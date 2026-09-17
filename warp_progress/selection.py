"""Pinned public-recipe selection, plus conservative local reviewed-task selection."""
import math
import random


def public_recipe_split(episodes, object_counts, shortest_fraction=.25, n_validation=20, clean_val_fraction=.15, seed=42):
    if not 0 < shortest_fraction <= 1 or not 0 < clean_val_fraction <= 1 or n_validation < 1:
        raise ValueError('Invalid paper selection parameters')
    groups = {}
    for ep in episodes:
        key = str(ep['episode_index'])
        if key not in object_counts:
            raise ValueError('Missing object count: ' + key)
        groups.setdefault(object_counts[key], []).append(ep)
    kept, strata = [], []
    for count in sorted(groups):
        rows = sorted(groups[count], key=lambda e:e['length'])
        n = max(1, round(shortest_fraction * len(rows)))
        kept.extend(rows[:n])
        strata.append(dict(object_count=count, total=len(rows), retained=n, longest_retained_frames=rows[n-1]['length']))
    n_pool = max(n_validation, round(clean_val_fraction * len(kept)))
    if n_pool >= len(kept):
        raise ValueError('Not enough episodes for an independent validation set')
    pool = sorted(kept, key=lambda e:e['length'])[:n_pool]
    random.Random(seed).shuffle(pool)
    val_ids = {e['episode_index'] for e in pool[:n_validation]}
    train = [e for e in kept if e['episode_index'] not in val_ids]
    random.Random(seed).shuffle(train)
    validation = pool[:n_validation]
    return train, validation, dict(strata=strata, shortest_fraction=shortest_fraction,
                                   clean_val_fraction=clean_val_fraction, validation_pool=n_pool, seed=seed,
                                   note='Matches pinned upstream filter_shortest_stratified then split_episodes_clean_val')
