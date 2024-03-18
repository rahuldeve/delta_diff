import random
from functools import wraps
from typing import Callable, Concatenate, ParamSpec

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
from rdkit.rdBase import BlockLogs


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize(smiles):
    with BlockLogs():
        # follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        mol = Chem.MolFromSmiles(smiles)  # type: ignore

        if mol is None:
            return None

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        # te = rdMolStandardize.TautomerEnumerator() # idem
        # taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        return Chem.MolToInchi(uncharged_parent_clean_mol)


Param = ParamSpec("Param")
BaseFunc = Callable[Concatenate[pd.DataFrame, Param], pd.DataFrame]
WrappedFunc = Callable[Concatenate[pd.DataFrame, Param], pd.DataFrame]


def parallelize(n_jobs: int = 4) -> Callable[[BaseFunc], WrappedFunc]:
    def _inner(f: BaseFunc) -> WrappedFunc:
        @wraps(f)
        def parallel_func(
            df: pd.DataFrame, *args: Param.args, **kwargs: Param.kwargs
        ) -> pd.DataFrame:
            num_chunks = len(df) // n_jobs
            tail_size = len(df) % num_chunks
            if tail_size == 0:
                chunks = np.vsplit(df, num_chunks)
            else:
                head = df.iloc[:-tail_size, :]
                tail = df.iloc[-tail_size:, :]
                chunks = np.vsplit(head, num_chunks) + [tail]

            jobs = Parallel(n_jobs=n_jobs)(
                delayed(f)(chunk, *args, **kwargs) for chunk in chunks
            )
            return pd.concat(jobs).reset_index(drop=True)  # type: ignore

        return parallel_func

    return _inner
