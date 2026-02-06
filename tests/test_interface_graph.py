from __future__ import annotations

from src.data.interface_graph import build_interface_patch
from src.data.pdb_utils import parse_pdb_text


def _toy_pdb() -> str:
    lines = []
    atom_id = 1
    for chain, y in [("A", 0.0), ("B", 6.0)]:
        for i in range(1, 60):
            x = i * 1.5
            for atom, off in [("N", -1.1), ("CA", 0.0), ("C", 1.2), ("O", 2.0), ("CB", 0.3)]:
                lines.append(
                    f"ATOM  {atom_id:5d} {atom:>4s} ALA {chain}{i:4d}    {x+off:8.3f}{y:8.3f}{0.0:8.3f}  1.00 20.00           C"
                )
                atom_id += 1
    lines.append("END")
    return "\n".join(lines)


def test_patch_builder_deterministic_and_max_nodes() -> None:
    s = parse_pdb_text(_toy_pdb())
    p1 = build_interface_patch(s, "AA10V", max_nodes=32)
    p2 = build_interface_patch(s, "AA10V", max_nodes=32)

    assert p1.aa_ids.shape[0] <= 32
    assert p1.edge_index.shape[0] == 2
    assert p1.edge_feat.shape[1] == 25
    assert (p1.aa_ids == p2.aa_ids).all()
    assert (p1.edge_index == p2.edge_index).all()


def test_edge_types_present_and_shapes() -> None:
    s = parse_pdb_text(_toy_pdb())
    p = build_interface_patch(s, "AA10V", max_nodes=40)
    assert p.edge_feat.shape[1] == 25
    edge_type_onehot = p.edge_feat[:, -3:]
    assert edge_type_onehot.sum(axis=1).min() >= 0.99
    assert edge_type_onehot.sum(axis=1).max() <= 1.01
