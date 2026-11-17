#!/usr/bin/env python3
"""Apply Sonnet-style truncated-normal initialization and whole-signal masking.

Put this file in the root of the grid-cells-torch repository and run:
    python apply_vision_trunc_normal.py

Makes backups of changed tracked files as *.py.before_vision_change.bak.
Does not touch Grid Network, LSTM, datasets, or checkpoints.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
VISION = ROOT / "grid_cells" / "vision.py"
CONFIG = ROOT / "config.py"

UPDATES = {
    VISION: [
        (
            "        self.fc = nn.Linear(128 * 4 * 4, embed_dim)\n\n"
            "    def forward(self, images: torch.Tensor) -> torch.Tensor:\n",
            "        self.fc = nn.Linear(128 * 4 * 4, embed_dim)\n"
            "        self._init_weights()\n\n"
            "    def _init_weights(self) -> None:\n"
            "        \"\"\"Initialize CNN and FC using a Sonnet-style truncated normal.\"\"\"\n"
            "        for layer in self.modules():\n"
            "            if isinstance(layer, (nn.Conv2d, nn.Linear)):\n"
            "                fan_in = layer.weight[0].numel()\n"
            "                std = fan_in ** -0.5\n"
            "                nn.init.trunc_normal_(layer.weight, mean=0.0, std=std,\n"
            "                                      a=-2.0 * std, b=2.0 * std)\n"
            "                if layer.bias is not None:\n"
            "                    nn.init.zeros_(layer.bias)\n\n"
            "    def forward(self, images: torch.Tensor) -> torch.Tensor:\n",
        ),
        (
            'mask_mode: str = "per_unit"',
            'mask_mode: str = "whole"',
        ),
        (
            "        self.hd_head = nn.Linear(embed_dim, hd_ensemble.n_cells)\n"
            "        self.mask_prob = mask_prob\n",
            "        self.hd_head = nn.Linear(embed_dim, hd_ensemble.n_cells)\n"
            "        for head in (self.place_head, self.hd_head):\n"
            "            std = head.in_features ** -0.5\n"
            "            nn.init.trunc_normal_(head.weight, mean=0.0, std=std,\n"
            "                                  a=-2.0 * std, b=2.0 * std)\n"
            "            nn.init.zeros_(head.bias)\n"
            "        self.mask_prob = mask_prob\n",
        ),
    ],
    CONFIG: [
        ('vision_mask_mode: str = "per_unit"',
         'vision_mask_mode: str = "whole"'),
    ],
}


def main() -> int:
    # Validate all changes before writing *any* file.
    prepared = {}
    for path, replacements in UPDATES.items():
        if not path.is_file():
            print(f"ERROR: Cannot find {path}", file=sys.stderr)
            print("Place this script in the root of grid-cells-torch.", file=sys.stderr)
            return 1
        original = path.read_text(encoding="utf-8")
        updated = original
        for before, after in replacements:
            count = updated.count(before)
            if count == 1:
                updated = updated.replace(before, after, 1)
            elif count == 0 and after in updated:
                pass  # Already applied; permit safe repeat invocation.
            else:
                print(f"ERROR: Expected code not found uniquely in {path} (count={count}).", file=sys.stderr)
                print("Nothing has been changed. Your local code may differ from GitHub main.", file=sys.stderr)
                return 1
        prepared[path] = (original, updated)

    for path, (original, updated) in prepared.items():
        if original == updated:
            print(f"Already up to date: {path.relative_to(ROOT)}")
            continue
        backup = path.with_name(path.name + ".before_vision_change.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(updated, encoding="utf-8")
        print(f"Updated: {path.relative_to(ROOT)}")
        print(f"Backup : {backup.relative_to(ROOT)}")

    print("Done. Vision cross-entropy loss still uses unmasked logits.")
    print("At inference / Grid Network input, the PC+HD predictions are jointly zeroed 95% of the time.")
    print("No existing checkpoints or Grid Network weights were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
