"""Dependency-light chemical formula parsing utilities."""

from __future__ import annotations

import math
import re
from collections import defaultdict


ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os "
    "Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md "
    "No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ELEMENT_SET = frozenset(ELEMENT_SYMBOLS)

_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|(?:\d+(?:\.\d*)?|\.\d+))")


class FormulaError(ValueError):
    pass


def _tokenize(formula: str) -> list[str]:
    compact = formula.replace(" ", "")
    if not compact:
        raise FormulaError("formula is empty")
    tokens = _TOKEN.findall(compact)
    if "".join(tokens) != compact:
        raise FormulaError("formula contains unsupported characters")
    return tokens


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a conventional formula, including nested parentheses and decimals."""

    tokens = _tokenize(formula)

    def parse_group(index: int, nested: bool) -> tuple[dict[str, float], int]:
        values: defaultdict[str, float] = defaultdict(float)
        while index < len(tokens):
            token = tokens[index]
            if token == ")":
                if not nested:
                    raise FormulaError("unmatched closing parenthesis")
                return dict(values), index + 1
            if token == "(":
                subgroup, index = parse_group(index + 1, True)
                multiplier = 1.0
                if index < len(tokens) and _is_number(tokens[index]):
                    multiplier = float(tokens[index])
                    index += 1
                _validate_count(multiplier)
                for symbol, count in subgroup.items():
                    values[symbol] += count * multiplier
                continue
            if _is_number(token):
                raise FormulaError("a coefficient must follow an element or group")
            if token not in ELEMENT_SET:
                raise FormulaError(f"unknown element symbol: {token}")
            index += 1
            count = 1.0
            if index < len(tokens) and _is_number(tokens[index]):
                count = float(tokens[index])
                index += 1
            _validate_count(count)
            values[token] += count
        if nested:
            raise FormulaError("unclosed parenthesis")
        return dict(values), index

    parsed, final_index = parse_group(0, False)
    if final_index != len(tokens) or not parsed:
        raise FormulaError("formula could not be fully parsed")
    return dict(sorted(parsed.items(), key=lambda item: ELEMENT_SYMBOLS.index(item[0])))


def _is_number(token: str) -> bool:
    return token[0].isdigit() or token[0] == "."


def _validate_count(count: float) -> None:
    if not math.isfinite(count) or count <= 0:
        raise FormulaError("stoichiometric coefficients must be finite and positive")


def molar_mass(composition: dict[str, float]) -> float | None:
    """Return average molar mass when RDKit's periodic table is installed."""

    try:
        from rdkit.Chem import GetPeriodicTable
    except ImportError:
        return None
    table = GetPeriodicTable()
    return sum(table.GetAtomicWeight(symbol) * count for symbol, count in composition.items())

