"""Preflop best response over public action histories and hidden equity ranges."""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

import rivercount as agent
from game import Action, Deal, Game, Street
from rl_policy import ResidualPolicy, state_key


@dataclass(frozen=True)
class Leaf:
    pot: int
    contributions: tuple
    winner: int | None
    allin: bool


@dataclass
class Decision:
    actor: int
    history: tuple
    obs: object
    children: dict


@dataclass(frozen=True)
class Settings:
    future_bets: float = 1.0
    position: float = .04
    margin: float = .02
    leak: float = .035
    min_raises: int = 0


@lru_cache(maxsize=1)
def betting_tree():
    """Construct the standard 40-blind public tree once, with button in seat 0."""
    lookup = {}

    def build(game, history):
        actor = game.to_act
        obs = game.observation(actor)
        node = Decision(actor, history, obs, {})
        lookup[history] = node
        for action in obs.legal_actions:
            if action == Action.FOLD:
                child = Leaf(game.pot, tuple(game.contributed), 1 - actor, False)
            elif action == Action.CHECK_CALL and obs.opp_stack == 0:
                # Avoid dealing or evaluating the placeholder board at an all-in.
                contrib = game.contributed.copy()
                contrib[actor] += min(obs.to_call, obs.my_stack)
                child = Leaf(sum(contrib), tuple(contrib), None, True)
            else:
                branch = game.clone()
                branch.apply(action)
                if branch.street != Street.PREFLOP:
                    child = Leaf(branch.pot, tuple(branch.contributed), None, branch.is_terminal)
                else:
                    child = build(branch, history + (action,))
            node.children[action] = child
        return node

    dummy = Deal((tuple(range(5)), tuple(range(5, 10))), tuple(range(10, 15)))
    return build(Game(dummy, 0), ()), lookup


def modeled_actions(obs, mirror, levels):
    """Vectorized preflop bets under the reference policy for possible equities."""
    exponent = math.log(agent._range_equity(obs, mirror, .5)) / math.log(.5)
    eq = levels ** exponent
    pot = obs.pot
    price = min(obs.to_call, obs.my_stack)
    after_call = pot + price
    behind = obs.my_stack - price
    _, aggression = agent._response_rates(mirror, obs.street, None)
    fold, reraise = agent._response_rates(mirror, obs.street, Action.RAISE_POT)
    actions = np.full(len(eq), int(Action.CHECK_CALL))
    if aggression > .86 and reraise > .80 and fold < .09:
        implied = .35 * min(behind, 2 * after_call)
        actions[price > eq * (after_call + implied)] = int(Action.FOLD)
    else:
        realization = (.035 + .06 * aggression + .018 * (not obs.is_button)) * min(1, behind / after_call)
        best = (eq - realization) * after_call - price
        if Action.FOLD in obs.legal_actions:
            actions[best < 0] = int(Action.FOLD)
            best = np.maximum(best, 0)
        for action, target in obs.raise_to.items():
            cost = target - obs.my_bet
            extra = cost - price
            fold, reraise = agent._response_rates(mirror, obs.street, action)
            if obs.raises_this_street > 0:
                fold *= .72
            called = np.maximum(0, eq - .19 * fold * (1 - eq) / max(.01, 1 - fold))
            value = fold * pot + (1 - fold) * (called * (after_call + 2 * extra) - cost)
            patience = .065 + (.15 * aggression if aggression > .70 else 0)
            if extra < behind and obs.raises_this_street + 1 < mirror["config"].raise_cap:
                patience = patience + .16 * reraise * (1 - eq)
            value -= extra * patience
            actions[value > best] = int(action)
            best = np.maximum(best, value)
    # Most public nodes have no learned residual at any equity. Querying every
    # hidden bin separately there would add thousands of identical misses.
    if isinstance(agent._POLICY, ResidualPolicy):
        sample_key = state_key(obs, mirror, Action.CHECK_CALL, .5)
        if sample_key is None:
            return actions.astype(np.int8)
        prefix = sample_key.rsplit("/", 2)[0] + "/"
        if not any(key.startswith(prefix) for key in agent._POLICY.actions):
            return actions.astype(np.int8)
    return np.fromiter((int(agent._POLICY.select(obs, mirror, Action(a), e))
                        for e, a in zip(levels, actions)), dtype=np.int8, count=len(levels))


def solve(mirror, table, settings):
    prior, matchup, centers = table["prior"], table["equity"], table["centers"]
    root, nodes = betting_tree()
    transitions = {}
    for history, node in nodes.items():
        acts = modeled_actions(node.obs, mirror, centers.ravel()).reshape(centers.shape)
        transitions[history] = {
            a: (1 - settings.leak) * (acts == a).mean(axis=1) + settings.leak / len(node.children)
            for a in node.children
        }
    predictions = {}
    for hero in (0, 1):
        def visit(node, weights):
            if isinstance(node, Leaf):
                if node.winner is not None:
                    return np.full(len(prior), node.pot * (node.winner == hero) - node.contributions[hero])
                share = (weights * matchup).sum(axis=1)
                behind = mirror["config"].starting_stack - node.contributions[hero]
                extra = 0 if node.allin else min(behind, settings.future_bets * node.pot)
                positional = settings.position * extra * (1 if hero == 0 else -1)
                return share * node.pot - node.contributions[hero] + (2*share - 1)*extra + positional
            if node.actor == hero:
                values = {a: visit(child, weights) for a, child in node.children.items()}
                predictions[node.history] = values
                return np.max(list(values.values()), axis=0)
            expectation = np.zeros(len(prior))
            for action, child in node.children.items():
                posterior = weights * transitions[node.history][action]
                probability = posterior.sum(axis=1)
                expectation += probability * visit(child, posterior / probability[:, None])
            return expectation
        visit(root, prior)
    return predictions


def new_half(config, memory):
    agent.new_half(config, memory)
    mirror = {}
    agent.new_half(config, mirror)
    memory.update(mirror=mirror, model_hits=0, model_trials=0, refreshed=-1000, solved={})


def _update_model_fit(history, hero_action, obs, result, memory):
    """Score predictions only when public showdown results reveal opponent cards."""
    if result.opp_hole is None:
        return
    _, nodes = betting_tree()
    q = agent.preflop_equity(result.opp_hole)
    prefix = ()
    for event in history:
        if event.street != Street.PREFLOP:
            break
        if not event.is_me:
            opponent_obs = nodes[prefix].obs
            base = agent._pick_action(opponent_obs, memory["mirror"], q)
            predicted = agent._POLICY.select(opponent_obs, memory["mirror"], base, q)
            memory["model_trials"] += 1
            memory["model_hits"] += predicted == event.action
        prefix += (event.action,)
    if obs.street == Street.PREFLOP and hero_action >= Action.RAISE_HALF:
        # The last opponent reply is delivered as a showdown after our all-in.
        opponent_obs = nodes[prefix + (hero_action,)].obs
        base = agent._pick_action(opponent_obs, memory["mirror"], q)
        predicted = agent._POLICY.select(opponent_obs, memory["mirror"], base, q)
        memory["model_trials"] += 1
        memory["model_hits"] += predicted == Action.CHECK_CALL


def on_hand_end(result, memory):
    """Mirror what our public actions let the opponent learn about our frequencies."""
    last = memory["last"]
    if last is not None and last[0].hand_index == result.hand_index:
        obs, final = last
        history = obs.action_history
        _update_model_fit(history, final, obs, result, memory)
        config = memory["config"]
        blinds = [config.small_blind, config.big_blind] if obs.is_button else [config.big_blind, config.small_blind]
        bets = blinds.copy()
        stacks = [config.starting_stack - b for b in blinds]
        previous, street, raises = None, Street.PREFLOP, 0
        for event in history:
            if street != event.street:
                previous, street, raises, bets = None, event.street, 0, [0, 0]
            seat = int(not event.is_me)
            if event.is_me:
                can_raise = stacks[1] > 0 and stacks[0] > max(0, bets[1] - bets[0]) and raises < config.raise_cap
                agent._record_response(memory["mirror"], street, previous, event.action, can_raise)
            stacks[seat] -= event.amount
            bets[seat] = event.street_total
            raises += event.action >= Action.RAISE_HALF
            previous = event.action
        faced = previous if obs.street == street else None
        agent._record_response(memory["mirror"], obs.street, faced, final, bool(obs.raise_to))
    agent.on_hand_end(result, memory)


def choose_action(obs, memory, table, settings):
    base = agent.choose_action(obs, memory)
    trials = memory["model_trials"]
    if (obs.street != Street.PREFLOP or obs.raises_this_street < settings.min_raises
            or trials < 12 or memory["model_hits"] / trials < .84):
        return base
    if obs.hand_index - memory["refreshed"] >= 64:
        memory["solved"] = solve(memory["mirror"], table, settings)
        memory["refreshed"] = obs.hand_index
    node = tuple(event.action for event in obs.action_history)
    values = memory["solved"][node]
    q = memory["equity"]
    closest = int(np.searchsorted(table["boundaries"], q))
    best = max(values, key=lambda a: values[a][closest])
    if values[best][closest] - values[base][closest] <= settings.margin * obs.pot:
        return base
    memory["last"] = obs, best
    return best
