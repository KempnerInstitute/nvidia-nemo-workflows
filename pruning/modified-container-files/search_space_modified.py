# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An extension of the DynamicSpace to handle searchable units and related utilities."""

import fnmatch
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from modelopt.torch.opt import RulesDict
from modelopt.torch.opt.dynamic import DynamicModule, DynamicSpace, _DMRegistryCls
from modelopt.torch.trace import Symbol, SymMap, analyze_symbols
from modelopt.torch.utils import random

from .registry import DMRegistry
from .traced_hp import TracedHp, TracedHpRegistry

__all__ = ["generate_search_space", "SearchSpace"]

SampleFunc = Union[random.FSample, dict[str, random.FSample]]


class SearchSpace(DynamicSpace):
    """Stub for search space interface."""

    def __init__(self, model: nn.Module, sym_map: Optional[SymMap] = None) -> None:
        """Initialize the search space from the model and its associated symbolic map."""
        super().__init__(model)
        self._sym_map = sym_map
        if self._sym_map:
            assert self._sym_map.model is self.model, "Model and sym_map's model must match."

    def _sym_to_hp(self, sym: Symbol) -> TracedHp:
        assert isinstance(self._sym_map, SymMap), "Model's sym_map was not provided!"
        for name, sym_ in self._sym_map.named_symbols():
            if sym is sym_:
                mod, _, hp_name = name.rpartition(".")
                mod = self.model.get_submodule(mod)
                assert isinstance(mod, DynamicModule)
                hp = mod.get_hparam(hp_name)
                assert isinstance(hp, TracedHp), f"Expected TracedHp, got {type(hp)}!"
                return hp
        raise KeyError(f"Symbol {sym} not found in sym_map!")

    def _should_be_converted(self, mod: nn.Module) -> bool:
        return self._sym_map is None or mod in self._sym_map

    def generate(self, rules: Optional[RulesDict] = None) -> None:
        """Generate the search space from the model + symbolic map."""
        # sanity checks
        assert isinstance(self._sym_map, SymMap), "Model's sym_map was not provided!"

        # 1. + 2. convert to dynamic module and apply rules
        mods_converted = self.convert_to_dynamic(rules, DMRegistry)

        # 3. iterate through searchable/constant symbols and generate hparams for each
        #    dependency tree.
        sym_to_hp_all: dict[Symbol, TracedHp] = {}
        for name, sym in self._sym_map.named_symbols(searchable=True, constant=True):
            mod_name, _, sym_name = name.rpartition(".")
            mod = self.model.get_submodule(mod_name)
            assert isinstance(mod, DynamicModule), f"Module {mod} must be a DynamicModule!"

            hp = TracedHpRegistry.initialize_from(sym, mod.get_hparam(sym_name))
            setattr(mod, sym_name, hp)
            sym_to_hp = hp.resolve_dependencies(sym, self._sym_to_hp)
            sym_to_hp_all.update(sym_to_hp)

        # 4. sanity check that we cover all symbols and hyperparameters
        # NOTE: symbols and hparams of the non-leaf modules will not be included in the search space
        symbols_all = dict(self._sym_map.named_symbols())
        assert set(sym_to_hp_all.keys()) == set(symbols_all.values()), "Missing symbols!"

        # 5. iterate through all dynamic modules and reassign hparams accordingly
        for mod_name, mod in mods_converted.items():
            for sym_name, sym in self._sym_map[mod].items():
                # set new hp
                setattr(mod, sym_name, sym_to_hp_all[sym])

    def sample(self, sample_func: SampleFunc = random.choice) -> dict[str, Any]:
        """Sample configurable hparams using the provided sample_func and return resulting config.

        Args:
            model: A configurable model that contains one or more DynamicModule(s).
            sample_func: A sampling function for hyperparameters. If a
                dict is provided, the key indicates a Unix shell-style wildcard pattern which is
                used to match the hyperparameter name. The default is random sampling corresponding
                to ``{"*": random.choice}``.

                .. note::

                    Later matches take precedence over earlier ones when a dict with wildcards is
                    provided.

        Returns:
            A dict of ``(parameter_name, choice)`` that specifies an active subnet.
        """
        # wrap sample_func into a dict if it's not already
        if not isinstance(sample_func, dict):
            sample_func = {"*": sample_func}

        # sample configurable hparams
        f_sample: Optional[random.FSample]
        for hp_name, hp in self.named_hparams(configurable=True):
            for pattern, f_sample in sample_func.items():
                if fnmatch.fnmatch(hp_name, pattern):
                    break
            else:
                f_sample = None
            if f_sample:
                hp.active = f_sample(hp.choices)

        return self.config()

    @torch.no_grad()
    def sort_parameters(self, hps_to_sort: Optional[set[str]] = None) -> None:
        """A graph propagation based parameter sorting algorithm.

        Args:
            hps_to_sort: A set of hparam names to sort. If not provided or empty, all hparams will be sorted.
        """
        # get config and set to max
        config = self.config()
        self.sample(sample_func=max)
        

        # sort configurable hparams
        for name, hp in self.named_hparams(configurable=True):
            hp_name = name.split(".")[-1]
            if hps_to_sort and hp_name not in hps_to_sort:
                continue
            importance = hp.importance

            # no sorting to be done (maybe because there is nothing to sort or it's unsortable)
            if importance is None:
                continue

            self._visualize(torch.cuda.current_device(), name, importance)

            # compute order from importance and enforce it
            order = torch.argsort(importance, descending=True)
            hp.enforce_order(order)

        # now that we have enforced an order we can force reassign all parameters/buffers!
        for _, mod in self.named_dynamic_modules():
            mod.force_assign()

        # go back to old config
        self.select(config)

    def export(self, dm_registry: _DMRegistryCls = DMRegistry) -> nn.Module:
        """Export with default registry."""
        return super().export(dm_registry)

    def print_summary(self, skipped_hparams: list[str] = ["kernel_size"]) -> None:
        """Print a summary of the search space."""
        print("\nSearch Space Summary:\n{:-^100}".format(""))
        hp_visited = set()  # Only highlight configurable hparams once
        for name, hp in self.named_hparams():
            if not any(name.endswith(s) for s in skipped_hparams):
                if len(hp.choices) > 8:
                    choices_to_print = hp.choices[:4] + ["..."] + hp.choices[-4:]
                else:
                    choices_to_print = hp.choices
                if hp.is_configurable and hp not in hp_visited:
                    hp_visited.add(hp)
                    prefix = "*"
                else:
                    prefix = " "
                print(f"{prefix} {name:80} {choices_to_print}")

        print("{:-^100}".format(""))

    
    def _visualize(self, device_id, layer_name, importance, save_dir="plots"):
        layer_number = self._get_layer_number(layer_name)
        name = self._get_clean_name(layer_name)
        importance_cpu = importance.cpu() 

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(importance_cpu)), importance_cpu.numpy())

        if layer_number is not None:
            title = f"Layer {layer_number}-{name} Importance"
        else:
            title = f"{name} Importance"
        plt.title(title)
        plt.xlabel(name)
        plt.ylabel("Importance Score")
        plt.grid(True)

        # Create directory with cuda device subfolder
        device_dir = os.path.join(save_dir, f"cuda{device_id}")
        os.makedirs(device_dir, exist_ok=True)

        # Format filename with lowercase and spaces to underscores
        filename = title.lower().replace(" ", "_") + ".png"
        filepath = os.path.join(device_dir, filename)

        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot to {filepath}")

        
    def _get_layer_number(self, name: str) -> int | None:
        parts = name.split('.')
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return int(parts[idx + 1])
        return None


    def _get_clean_name(self, name: str) -> str:
        parts = name.split('.')
        layer_num = self._get_layer_number(name)

        if layer_num is not None:
            idx = parts.index("layers")
            sub_parts = parts[idx + 2:]  # parts after layers.<num>
        else:
            sub_parts = parts[-2:] if len(parts) >= 2 else parts[-1:]

        param_name_raw = ".".join(sub_parts)

        name_map = {
            "self_attention.num_heads_per_group": "Attention Head",
            "self_attention.num_query_groups": "Attention Query Group",
            "mlp.ffn_hidden_size": "Neuron",
            "module.hidden_size": "Embedding Channel",
        }

        return name_map.get(param_name_raw, param_name_raw.replace("_", " ").title())




def generate_search_space(model: nn.Module, rules: Optional[RulesDict] = None) -> SearchSpace:
    """Patch model with dynamic units and generate/return the search space.

    Args:
        model: The model to patched for which we want to generate the search space.
        rules: An optional dict specifying the rules used during the search space generation.
            If not provided, the largest possible search space is generated. An example rule:

            .. code-block:: python

                rules = {
                    "nn.Sequential": {"min_depth": 2},
                    "nn.Conv2d": {
                        "*": {
                            "channels_ratio": [1],
                            "channel_divisor": 16,
                            "kernel_size": [],
                        },
                        "backbone.stages.[1-5]*.spatial_conv*": {
                            "channels_ratio": [0.334, 0.5, 0.667, 1],
                        },
                        "det_blocks.*.spatial_conv*": {
                            "channels_ratio": [0.667, 1],
                        },
                    },
                    "nn.Linear": {},  # uses default arguments for Linear
                    "nn.BatchNorm2d": {},
                    "nn.SyncBatchNorm": {},
                }

            .. note::

                When rules are provided, the search space generation is restricted to the rules that
                are provided. Other module types will not be included in the search space.

            .. note::

                Rules with ``None``, e.g., ``{"nn.Sequential": None}``, will also cause the
                conversion to be skipped just like when no rules are provided for the layer type.

            .. note::

                Instead of specifying the module type as a string, the module type can also be
                specified as a key directly, e.g., ``{nn.Sequential: None}``.

    Returns:
        The patched model with dynamic units that describe the resulting search space.

    Generally, the search space generation follows the following outline of steps:

        1. Analyze the model to identify the dynamic units that can be patched via modelopt.torch.trace.
        2. Patch the model with the dynamic units accordingly.
        3. Generate a consistent search space from the patched model by ensuring consistency between
           dynamic units according to the symbolic map obtained from the model analysis.

    Upon return of the model, the search space can be analyzed and modified via the utilities
    provided in ``modelopt.torch.dynamic.SearchSpace``.
    """
    sym_map = analyze_symbols(model)
    search_space = SearchSpace(model, sym_map)
    search_space.generate(rules=rules)
    return search_space
