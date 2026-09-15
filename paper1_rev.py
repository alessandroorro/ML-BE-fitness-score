
import itertools
import ast
import re
import contextlib
import os
import pathlib

os.environ["GSK_RENDERER"] = "cairo"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"



import matplotlib as mpl

mpl.use("TkAgg")  # oppure "Qt5Agg"
import matplotlib.pylab as plt
if not getattr(plt.Figure, "to_clipboard", None):
    plt.Figure.to_clipboard = lambda self: None

import joblib



import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)


# fix dataframe-utils
if not getattr(pd.DataFrame, "to_word_clipboard", None):
    pd.DataFrame.to_word_clipboard = lambda self: None



from Bio.SeqUtils import MeltingTemp

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.inspection

import statsmodels.api as sm

import lightgbm

import xgboost

import catboost

import shap

from tqdm import tqdm

from scipy.stats import wilcoxon

cell_lines = 'A375', 'MELJUSO', 'OVCAR8', 'HAP1', 'HA1E'

try:
    from utils import ExperimentS
except ImportError:
    class ExperimentS:
        def __init__(self, storage_path, force, n_inputs, n_replicates, n_folds, conditions):
            self.storage_path = pathlib.Path(storage_path).with_suffix('.log')
            self.force = force
            self.n_inputs = n_inputs
            self.n_replicates = n_replicates
            self.n_folds = n_folds
            self.conditions = sorted(conditions)

        def get_scores(self, replicate, fold, conditions):
            conds = sorted(conditions)
            kw = '-'.join(f"{k}{conds[k]}" for k in sorted(conds.keys()))
            ipath = pathlib.Path(f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}.scores.npy")
            with ipath.open("rb") as f:
                return np.load(f)

        def register_scores(self, replicate, fold, conditions, scores):
            conds = sorted(conditions)
            kw = '-'.join(f"{k}{conds[k]}" for k in sorted(conds.keys()))
            opath = pathlib.Path(f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}.scores.npy")
            with opath.open('wb') as f:
                np.save(f, scores)







def encode_onehot(seq):
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq]).flatten()


def buildXf(dataset, Xf=None, context=None, cell_line='A375', use_gene_strand=True, use_ruleset2=False,
            use_efficiency=False):
    if Xf is None:
        Xf = {}
        # mutation_bins = sorted(dataset['Mutation bin'].unique())
        
        # ['Intron', 'Missense', 'No edits', 'Nonsense', 'Silent', 'Splice site', 'UTR']
        mutation_bins = [
            'Intron',
            'Missense',
            'No edits',
            'Nonsense',
            'Silent',
            'Splice site',
            'UTR',
        ]
        # mutation_bins = {k: i for i, k in enumerate(mutation_bins)}
        Xf['mutation_bin'] = OneHotEncoder(
            sparse_output=False,
            categories=[mutation_bins],
        ).fit_transform(dataset['Mutation bin'].values.reshape(-1, +1))
        # Xf['mutation_bin'] = OneHotEncoder(sparse_output=False, categories=[mutation_bins]).fit_transform(
        #     dataset['Mutation bin'].apply(lambda x: mutation_bins.index(x)).values.reshape(-1, +1)
        # )
        
        if use_gene_strand is True:
            Xf['strand'] = np.stack(
                dataset[['Gene strand', 'sgRNA strand']].apply(
                    lambda row: [(row['Gene strand'] + 1) / 2, 1 if row['sgRNA strand'] == 'sense' else 0],
                    axis=1
                ).to_list()
            )
        else:
            Xf['strand'] = np.stack(
                dataset[['sgRNA strand']].apply(
                    lambda row: [1 if row['sgRNA strand'] == 'sense' else 0],
                    axis=1
                ).to_list()
            )
        
        if use_efficiency is True:
            Xf['efficiency'] = \
                dataset[['efficiency:behive:BE4', 'efficiency:bedict:BE4max', 'efficiency:deepbe:NG']].values
        
        if use_ruleset2 is True:
            Xf['RuleSet2'] = \
                dataset[['On-target efficacy score', ]].values
        
        Xf['domains'] = \
            dataset['domains:pfam'].fillna('').apply(lambda x: 0 if x == '' else 1).values.reshape(-1, +1)
        
        Xf['depmap'] = dataset[[f'DepMap:{cell_line}', ]].values
        # Xf['depmap'] = dataset[['DepMap:A375', 'DepMap:HA1E', 'DepMap:HAP1', 'DepMap:MELJUSO', 'DepMap:OVCAR8']].values
    
    seq_encoding = encode_onehot
    
    if context is not None:
        if f'melt{context}' not in Xf:
            Xf[f'sequence{context}'] = \
                np.array([seq_encoding(seq) for seq in dataset[f"context{context}"].values])
            Xf[f'melt{context}'] = \
                dataset[f"context{context}"].apply(lambda seq: MeltingTemp.Tm_NN(seq)).values.reshape(-1, +1)
            Xf[f'gc{context}'] = \
                dataset[f"context{context}"].apply(lambda seq: (seq.count('G') + seq.count('C')) / len(seq)) \
                    .values.reshape(-1, +1)
    
    return Xf


def buildX(Xf, context, featureset):
    X = []
    F = []
    G = {}
    for feature in ('sequence', 'melt', 'gc'):
        if feature in featureset:
            X.append(Xf[f'{feature}{context}'])
            
            if feature == 'sequence':
                if context == 20:
                    offset = 0
                elif context == 30:
                    offset = 4
                elif context == 40:
                    offset = 9
                elif context == 50:
                    offset = 20
                for i in range(0, Xf[f'{feature}{context}'].shape[1], 4):
                    i = i // 4
                    F.append(f'seq:A:{i + offset + 1}')
                    F.append(f'seq:C:{i + offset + 1}')
                    F.append(f'seq:G:{i + offset + 1}')
                    F.append(f'seq:T:{i + offset + 1}')
                    
                    G[f'seq:A:{i + offset + 1}'] = [f'seq:{i + offset + 1}', 'seq']
                    G[f'seq:C:{i + offset + 1}'] = [f'seq:{i + offset + 1}', 'seq']
                    G[f'seq:G:{i + offset + 1}'] = [f'seq:{i + offset + 1}', 'seq']
                    G[f'seq:T:{i + offset + 1}'] = [f'seq:{i + offset + 1}', 'seq']
                
                for i in range(0, Xf[f'{feature}{context}'].shape[1], 4):
                    i = i // 4
                    # P.append(f'seq:{i+offset+1}')
                    # P.append(f'seq:{i+offset+1}')
                    # P.append(f'seq:{i+offset+1}')
                    # P.append(f'seq:{i+offset+1}')
            else:
                F.append(feature)
                # P.append(feature)
            
            # G.extend(Xf[f'{feature}{context}'].shape[1] * [feature])
    
    if 'RuleSet2' in featureset:
        X.append(Xf['RuleSet2'])
        F.append('RuleSet2')
    
    if 'efficiency' in featureset:
        X.append(Xf['efficiency'])
        F.append('efficiency')
    
    # for feature in ['efficiency:bedict:BE4max', 'efficiency:behive:BE4', 'efficiency:deepbe:NG']:
    #     if feature in featureset:
    #         X.append(Xf[feature])
    #         F.append(feature)
    #
    
    # Xf.keys()
    
    for feature in ('strand', 'depmap', 'domains', 'mutation_bin'):
        if feature not in featureset:
            continue
        
        X.append(Xf[feature])
        # G.extend(Xf[f'{feature}'].shape[1] * [feature])
        if feature == 'mutation_bin':
            F.extend(['Intron', 'Missense', 'No edits', 'Nonsense', 'Silent', 'Splice site', 'UTR'])
            G['Intron'] = 'Mutation'
            G['Missense'] = 'Mutation'
            G['No edits'] = 'Mutation'
            G['Nonsense'] = 'Mutation'
            G['Silent'] = 'Mutation'
            G['Splice site'] = 'Mutation'
            G['UTR'] = 'Mutation'
            
            # P.extend(['Intron', 'Missense', 'No edits', 'Nonsense', 'Silent', 'Splice site', 'UTR'])
        else:
            F.append(feature)
            # P.append(feature)
    
    return np.concatenate(X, axis=1), np.array(F), G


def train(
    training_set="analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx",
    
    model='GaussianNB',
    # model='LGBMClassifier',
    cell_line='A375',
    context_length=20,
    target="BE39",  # or CAS9
    use_gene_strand=False,
    use_efficiency=False,
    use_ruleset2=False,
    
    force_training=False,
    do_shap=False,
    do_perm=False
):
    """
    funzione principale che calcolata tutto salvando in file parziali
    
    do_shap e do_perm servono per specificare che devono essere salvati anche i risultati di feature importance
    
    """
    
    if isinstance(training_set, str):
        dataset = pd.read_excel(training_set, engine='openpyxl')
    else:
        dataset = training_set
    
    # dataset['RuleSet2'] = dataset['On-target efficacy score']
    
    # dataset = pd.read_excel(training_set, engine='openpyxl')
    genes = dataset['Gene symbol'].unique()
    if target == 'BE39':
        Y = (dataset[f"BE39:{cell_line}:zscore"].values < -2).astype(np.int8)
    else:
        raise ValueError(f"target {target} not supported")
    
    if use_efficiency or use_ruleset2:
        experimentS = ExperimentS(
            f"analysis/2024-02-paper1/results/paper1_rev/exp2",
            force=False,
            n_inputs=len(dataset),
            n_replicates=1,
            n_folds=len(dataset),
            conditions=[
                "model", "cell_line", "context_length", "target", "use_gene_strand", "max_depth"
            ]
        )
    else:
        experimentS = ExperimentS(
            f"analysis/2024-02-paper1/results/paper1_rev/exp_new",
            force=False,
            n_inputs=len(dataset),
            n_replicates=1,
            n_folds=len(dataset),
            conditions=[
                "model", "cell_line", "context_length", "target", "use_gene_strand", "max_depth"
            ]
        )
    
    condition = dict(
        use_gene_strand=use_gene_strand,
        context_length=context_length,
        model=model,
        cell_line=cell_line,
        max_depth=-1,
        target=target,
    )
    
    if model == 'GaussianNB':
        conditionlist = [
            condition.copy()
        ]
    elif model == 'DecisionTree':
        conditionlist = [
            {**condition.copy(), 'max_depth': max_depth}
            for max_depth in [None, 50, 15, 5]
        ]
    elif model == 'RandomForest':
        conditionlist = [
            {**condition.copy(), 'max_depth': max_depth}
            for max_depth in [None, 50, 15, 5]
        ]
    elif model == 'XGBClassifier':
        conditionlist = [
            {**condition.copy(), 'max_depth': max_depth}
            for max_depth in [12, 6, 3]
        ]
    elif model == 'LGBMClassifier':
        conditionlist = [
            {**condition.copy(), 'max_depth': max_depth}
            for max_depth in [-1, 4, 6]
        ]
    elif model == 'CatBoostClassifier':
        conditionlist = [
            {**condition.copy(), 'max_depth': max_depth}
            for max_depth in [2, 4, 6]
        ]
    else:
        raise ValueError(f"model {model} not supported")
    
    Xf = buildXf(
        dataset,
        cell_line=cell_line,
        use_gene_strand=use_gene_strand,
        use_efficiency=use_efficiency,
        use_ruleset2=use_ruleset2
    )
    Xf = buildXf(
        dataset,
        Xf,
        context=context_length,
        use_gene_strand=use_gene_strand,
        use_efficiency=use_efficiency,
        use_ruleset2=use_ruleset2
    )
    Xf
    featureset = ['sequence', 'strand', 'depmap', 'domains', 'melt', 'gc', 'mutation_bin']
    if use_ruleset2:
        featureset.append('RuleSet2')
    if use_efficiency:
        featureset.extend(['efficiency'])
    
    Xf['depmap']
    
    X, F, G = buildX(
        Xf,
        context=context_length,
        featureset=featureset
    )
    Xf['depmap'].sum() == X[:, 83].sum()
    np.where(F == 'depmap')
    
    'RuleSet2' in F
    'efficiency' in F
    np.where(F == '')
    
    # X.shape, F.shape, G.shape
    
    def model2clf(model, conditions):
        if model == 'GaussianNB':
            clf = GaussianNB()
        elif model == 'DecisionTree':
            clf = DecisionTreeClassifier(max_depth=conditions['max_depth'])
        elif model == 'RandomForest':
            clf = RandomForestClassifier(max_depth=conditions['max_depth'])
        elif model == 'XGBClassifier':
            clf = xgboost.XGBClassifier(max_depth=conditions['max_depth'])
        elif model == 'CatBoostClassifier':
            clf = catboost.CatBoostClassifier(max_depth=conditions['max_depth'], verbose=0)
        elif model == 'LGBMClassifier':
            clf = lightgbm.LGBMClassifier(max_depth=conditions['max_depth'], verbose=-1)
        else:
            raise ValueError(f"model {model} not supported")
        return clf
    
    for conditions in conditionlist:
        Spred = experimentS.get_scores(replicate=1, fold=None, conditions=conditions)
        # print("Spred is None", Spred is None)
        print(f"conditions:{conditions}")
        if Spred is None or force_training or do_perm or do_shap:
            Spred = np.zeros(len(X))
            for ifold, gene in tqdm(enumerate(genes), total=len(genes)):
                
                sel_tr = (dataset['Gene symbol'] != gene).values
                sel_va = (dataset['Gene symbol'] == gene).values
                
                Xtr = X[sel_tr]
                Xva = X[sel_va]
                Ytr = Y[sel_tr]
                Yva = Y[sel_va]
                
                clf = model2clf(model, conditions)
                
                conds = conditions.copy()
                conds['use_efficiency'] = use_efficiency
                conds['use_ruleset2'] = use_ruleset2
                kw = '-'.join(f"{k}{conds[k]}" for k in sorted(conds.keys()))
                
                model_file = pathlib.Path(
                    f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}-gene{gene}.joblib")

                if force_training or not model_file.exists():
                    clf.fit(Xtr, Ytr)
                    joblib.dump(clf, model_file)
                else:
                    clf = joblib.load(model_file)
                
                spred_va = clf.predict_proba(Xva)[:, 1]
                Spred[sel_va] = spred_va
                
                # print(kw)
                
                pifile = pathlib.Path(
                    f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}-gene{gene}.PI.joblib")
                if do_perm is True and not pifile.exists():
                    PI = sklearn.inspection.permutation_importance(
                        clf, Xva, Yva, scoring="roc_auc", n_repeats=10,
                        n_jobs=None, random_state=None
                    )

                    i = np.where(F=='depmap')[0][0]
                    PI['importances_mean'][i]
                    Xva[:, 83]
                    
                    joblib.dump(PI, pifile)
                    
                    # PI.importances
                    # PI.importances_mean.shape
                    # PI.importances_std.shape
                    # PI.importances_mean
                elif do_perm:
                    PI = joblib.load(pifile)
                
                svfile = pathlib.Path(
                    f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}-gene{gene}.SV.joblib")
                if do_shap is True and not svfile.exists():
                    if model in ('DecisionTreeClassifier', 'RandomForestClassifier',
                                 'XGBClassifier', 'CatBoostClassifier', 'LGBMClassifier'):
                        
                        explainer = shap.TreeExplainer(clf)
                        SV = explainer.shap_values(Xva)
                    else:
                        explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(Xtr, 100))
                        SV = explainer.shap_values(Xva, silent=True)
                    joblib.dump(SV, svfile)
                    
                    # import matplotlib.pyplot as plt
                    # plt.figure(figsize=(10, 6))
                    # shap.summary_plot(SV[:, :, 1], Xva, feature_names=F)
                    # plt.show()
                    
                    # GaussianNB (241, 92, 2)
                    # print("SV.shape", SV.shape)
                    # Xva.shape
                    # Xva.shape
                    # SV.mean(0)[:, 0].max()
                elif do_shap:
                    SV = joblib.load(svfile)
                

                # print("SV.shape", SV.shape)
                # print("PI.shape", PI.importances.shape)
                # print("PI.shape", PI.importances_mean.shape)
                # print("PI.shape", PI.importances_std.shape)
            
            model_file = pathlib.Path(f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}.joblib")
            if force_training or not model_file.exists():
                clf = model2clf(model, conditions)
                clf.fit(X, Y)
                joblib.dump(clf, model_file)
            else:
                clf = joblib.load(model_file)
            experimentS.register_scores(replicate=1, fold=None, conditions=conditions, scores=Spred)
        
        AUC, PR = roc_auc_score(Y, Spred), average_precision_score(Y, Spred)
        print(f"AUC:{AUC}, PR:{PR}")
        print()


def validate(
    validation_set="analysis/2024-02-paper1/dataset/coelho/dataset-efficiency-depmap-domains.xlsx",
    model='GaussianNB',
    cell_line='A375',
    context_length=20,
    target="BE39",  # or CAS9
    use_gene_strand=False,
    zscore_threshold=-2
):
    if isinstance(validation_set, str):
        dataset = pd.read_excel(validation_set, engine='openpyxl')
    else:
        dataset = validation_set
    # dataset = pd.read_excel(validation_set, engine='openpyxl')
    
    conditions = {
        'use_gene_strand': use_gene_strand,
        'context_length': context_length,
        # 'model': 'CatBoostClassifier',
        'model': model,
        'cell_line': cell_line,
        'max_depth': None,
        'target': target
    }
    
    kw = '-'.join(f"{k}{conditions[k]}" for k in sorted(conditions.keys()))
    clf = joblib.load(f"analysis/2024-02-paper1/results/paper1_rev/classifier-{kw}.joblib")
    
    Xf = buildXf(dataset, use_gene_strand=use_gene_strand)
    Xf = buildXf(dataset, Xf, context=context_length)
    X, F, G = buildX(
        Xf,
        context=context_length,
        featureset=('sequence', 'strand', 'depmap', 'domains', 'melt', 'gc', 'mutation_bin')
    )
    
    if target == 'BE39':
        Y = (dataset[f"BE39:HT29:zscore"].values < zscore_threshold).astype(np.int8)
    else:
        raise ValueError(f"target {target} not supported")
    
    Spred = clf.predict_proba(X)[:, 1]
    AUC = roc_auc_score(Y, Spred)
    
    return Y, Spred


def comparison_be39_vs_cas9_old():
    dataset = "analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx"
    dataset = pd.read_excel(dataset)
    
    dataset['delta:A375:zscore'] = dataset['BE39:A375:zscore'] - dataset['CAS9:A375:zscore']
    dataset['delta:MELJUSO:zscore'] = dataset['BE39:MELJUSO:zscore'] - dataset['CAS9:MELJUSO:zscore']
    dataset['n_edits'] = dataset['Nucleotide edits'].apply(ast.literal_eval).apply(lambda x: 0 if x == [''] else len(x))
    dataset['has_edit'] = dataset['n_edits'] > 0
    dataset['sgRNA strand'] = dataset['sgRNA strand'].apply(lambda x: 1 if x == 'sense' else -1)
    
    # prima estrai le posizioni
    def extract_positions(edits):
        if edits == [''] or edits == '':
            return []
        return [int(re.findall(r'\d+', e)[0]) for e in edits]
    
    dataset['edit_positions'] = dataset['Nucleotide edits'].apply(ast.literal_eval).apply(extract_positions)
    mut_bins = dataset['Mutation bin'].unique()
    for m in mut_bins:
        dataset['has_mutation:' + str(m)] = dataset['Mutation bin'] == m
    
    for pos in range(4, 9):  # posizioni 4,5,6,7,8
        dataset[f'has_edit_pos{pos}'] = dataset['edit_positions'].apply(lambda x: pos in x)
    for pos in range(1, 21):  # posizioni 1,2,...,20
        dataset[f'has_edit2_pos{pos}'] = dataset['sgRNA sequence'].apply(lambda seq: seq[pos - 1] == 'C')
    
    # group1 = dataset[(dataset['BE39:A375:zscore'] < -2) & (dataset['CAS9:A375:zscore'] > -0.5)]
    # group2 = dataset[(dataset['BE39:A375:zscore'] > -.5) & (dataset['CAS9:A375:zscore'] < -2)]
    # group1
    # group2
    # dataset['group'] = 'other'
    # dataset.loc[group1.index, 'group'] = 'BE_only'
    # dataset.loc[group2.index, 'group'] = 'CAS9_only'
    #
    #
    # cell_line = 'A375'
    # g1 = group1[f'DepMap:{cell_line}'].dropna()
    # g2 = group2[f'DepMap:{cell_line}'].dropna()
    
    with contextlib.nullcontext():
        groups = {
            # 'Predictors': ['Be-Dict', 'Be-Hive', 'DeepBE'],
            'No edits': ['No edits'],
            # 'RuleSet2': ['RuleSet2'],
            'Variant effect': ['Silent', 'Missense', 'Nonsense', 'Splice site'],
            'C in editing window': ['pos4', 'pos5', 'pos6', 'pos7', 'pos8'],
            'C outside editing window': ['pos1', 'pos2', 'pos9', 'pos10', 'pos11', 'pos12', 'pos13', 'pos14'],
            'C in seed region': ['pos15', 'pos16', 'pos17', 'pos18', 'pos19', 'pos20'],
            'DepMap': ['DepMap'],
            # 'Gene strand': ['Gene strand'],
        }
        
        # invertiamo la mappa: feature -> gruppo
        mapping = {feat: grp for grp, feats in groups.items() for feat in feats}
        coefficients = {}
        for cell_line in ('A375', 'MELJUSO'):
            dataset.columns
            # X = dataset[['DepMap:A375', 'sgRNA strand', 'Gene strand'] + [f'has_edit_pos{pos}' for pos in range(4, 9)]].astype(float)
            X = dataset[
                [f'DepMap:{cell_line}'] +
                ['sgRNA strand', 'Gene strand'] +
                # ["efficiency:bedict:BE4max", "efficiency:behive:BE4", "efficiency:deepbe:NG"] +
                # ["RuleSet2", ] +
                [f"has_mutation:{m}" for m in mut_bins] +
                [f'has_edit2_pos{pos}' for pos in range(1, 21)]
                ].astype(float)
            X = sm.add_constant(X)
            y = dataset[f'delta:{cell_line}:zscore']
            model = sm.OLS(y, X, missing='drop').fit()
            model.summary()
            model.pvalues
            model.params
            coefs = model.params.copy()
            coefs = coefs.drop('const')
            coefs = coefs.drop('sgRNA strand')
            coefs = coefs.rename(index=lambda x: x.replace('has_edit2_', '') if 'has_edit2_' in x else x)
            coefs = coefs.rename(index=lambda x: x.replace('has_mutation:', '') if 'has_mutation' in x else x)
            coefs = coefs.rename(index={f'DepMap:{cell_line}': 'DepMap'})
            coefs = coefs.rename(index={'efficiency:bedict:BE4max': 'Be-Dict', 'efficiency:behive:BE4': 'Be-Hive',
                                        'efficiency:deepbe:NG': 'DeepBE'})
            print(coefs)
            
            # coefs = coefs.groupby(mapping).sum().reindex(groups.keys())
            coefficients[cell_line] = coefs
            # plt.figure(figsize=(10, 6))
        
        coef_df = pd.concat(coefficients, axis=1).iloc[::-1]
        coef_df_grouped = coef_df.groupby(mapping).sum().reindex(groups.keys())
        fig, ax = plt.subplots(figsize=(10, 12))
        coef_df_grouped.iloc[::-1].plot(kind='barh', ax=ax)
        ax.axvline(0, linewidth=0.8)  # linea verticale a 0
        ax.set_xlabel('Coefficient')
        ax.set_title('Feature Effects on Δz-score (BE39 − CAS9) across cell lines')
        plt.tight_layout()
        plt.show()
        plt.savefig("analysis/2024-02-paper1/results/paper1_rev/comparison_be39_vs_cas9.png", dpi=300)
        


def internal_validation():
    training_set = pd.read_excel(
        "analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx",
        engine='openpyxl'
    )
    
    
    context_lengths = [20, 30, 40, 50]
    # context_lengths = [20]
    
    for context_length in context_lengths:
        for cell_line in cell_lines:
            # print(cell_line, context_length)
            
            train(
                training_set=training_set,
                model='CatBoostClassifier',
                # model='LGBMClassifier',
                # model='XGBClassifier',
                # model='RandomForest',
                # model='GaussianNB',
                # model='DecisionTree',
                cell_line=cell_line,
                context_length=context_length,
                target="BE39",  # or CAS9
                use_gene_strand=False,
                force_training=False,
                use_efficiency=False,
                use_ruleset2=False,
                do_shap=True,
                do_perm=True,
            )
            pass
pass



def result_tables():
    """
    mettiamo nelle tabelle le metriche di performance per ogni cell line e per ogni contesto lunghezza.
    """
    
    experimentS1 = ExperimentS(f"analysis/2024-02-paper1/results/paper1_rev/exp")
    
    experimentS2 = ExperimentS(f"analysis/2024-02-paper1/results/paper1_rev/exp2")
    
    experimentS = experimentS1
    
    dataset = pd.read_excel(
        "analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx",
        engine='openpyxl'
    )
    
    models = [
        'CatBoostClassifier',
        'LGBMClassifier',
        'XGBClassifier',
        'RandomForest',
        'GaussianNB',
        'DecisionTree',
    ]
    
    model2maxdepths = {
        'GaussianNB': [-1],
        'DecisionTree': [None, 50, 15, 5],
        'RandomForest': [None, 50, 15, 5],
        'XGBClassifier': [12, 6, 3],
        'LGBMClassifier': [-1, 4, 6],
        'CatBoostClassifier': [2, 4, 6],
    }
    cell_lines = 'A375', 'MELJUSO', 'OVCAR8', 'HAP1', 'HA1E'
    context_lengths = [20, 30, 40, 50]
    
    # %%
    report = []
    
    for cell_line in tqdm(cell_lines):
        Y = (dataset[f"BE39:{cell_line}:zscore"].values < -2).astype(np.int8)
        for model in models:
            for context_length in context_lengths:
                for max_depth in model2maxdepths[model]:
                    conditions = dict(
                        cell_line=cell_line,
                        context_length=context_length,
                        model=model,
                        target="BE39",
                        use_gene_strand=False,
                        max_depth=max_depth
                    )
                    
                    Spred = experimentS.get_scores(replicate=1, fold=None, conditions=conditions)
                    if Spred is None:
                        print('Spred is None for conditions', conditions)
                        continue
                    AUC = roc_auc_score(Y, Spred)
                    PR = average_precision_score(Y, Spred)
                    MCC = matthews_corrcoef(Y, Spred.round())
                    report.append(dict(
                        cell_line=cell_line,
                        model=model,
                        context_length=context_length,
                        AUC=AUC,
                        PR=PR,
                        max_depth=max_depth,
                    ))
                    
                    # print(f"AUC: {AUC:.3f}, PR: {PR:.3f}, MCC: {MCC:.3f}")
    report = pd.DataFrame(report)
    report.iloc[0]
    # %%
    
    #
    # summary complessiva
    #
    summary = report.groupby(['cell_line', 'model']).agg({
        'AUC': ['mean', 'std', 'min', 'max'],
        'PR': ['mean', 'std', 'min', 'max']
    })
    
    # Rinominiamo per chiarezza (opzionale, ma utile per il paper)
    summary.columns = ['_'.join(col) for col in summary.columns.values]
    
    # Creiamo le colonne formattate come richiesto
    summary['AUC_str'] = (summary['AUC_mean'] * 100).round(2).astype(str) + " ± " + (summary['AUC_std'] * 100).round(
        2).astype(str)
    summary['PR_str'] = (summary['PR_mean'] * 100).round(2).astype(str) + " ± " + (summary['PR_std'] * 100).round(
        2).astype(str)
    # summary.[summary.reset_index()['model'].isin(["CatBoostClassifier", "LGBMClassifier", "RandomForest", "XGBClassifier"])]['AUC_mean'].min()
    
    # Se vuoi vedere solo le colonne utili
    summary_report = summary[['AUC_str', 'PR_str', 'AUC_max', 'PR_max']].reset_index()
    
    summary_report['model'] = pd.Categorical(summary_report['model'], categories=models, ordered=True)
    
    summary_report[['cell_line', 'model', 'AUC_str']].set_index(['model', 'cell_line']).unstack(
        level='cell_line').to_word_clipboard()
    summary_report[['cell_line', 'model', 'PR_str']].set_index(['model', 'cell_line']).unstack(
        level='cell_line').to_word_clipboard()
    
    #
    # questo è il risultato mediato su tutto
    #
    summary_report
    
    #
    # se invece prendo solo context=20
    #
    # summary = report[report['context_length']==50].groupby(['cell_line', 'model']).agg({
    #     'AUC': ['mean', 'std', 'min', 'max'],
    #     'PR': ['mean', 'std', 'min', 'max']
    # })
    
    
    
    summary_report.iloc[0]
    
    summary_report.iloc[0]
    
    # per il revisore
    report_stats = report.groupby(['model']).agg(
        mean_auc=('AUC', 'mean'),
        std_auc=('AUC', 'std'),
        min_auc=('AUC', 'min'),
        max_auc=('AUC', 'max'),
        mean_pr=('PR', 'mean'),
        std_pr=('PR', 'std'),
        min_pr=('PR', 'min'),
        max_pr=('PR', 'max')
    )
    
    # Calcoliamo la massima deviazione percentuale in assoluto dalla media
    report_stats['max_abs_diff_auc'] = np.maximum(
        (report_stats['mean_auc'] - report_stats['min_auc']).abs(),
        (report_stats['max_auc'] - report_stats['mean_auc']).abs()
    )
    report_stats['max_abs_diff_pr'] = np.maximum(
        (report_stats['mean_pr'] - report_stats['min_pr']).abs(),
        (report_stats['max_pr'] - report_stats['mean_pr']).abs()
    )
    report_stats['max_pct_deviation_auc'] = (report_stats['max_abs_diff_auc'] / report_stats['mean_auc']) * 100
    report_stats['max_pct_deviation_pr'] = (report_stats['max_abs_diff_pr'] / report_stats['mean_pr']) * 100
    
    print(report_stats[['mean_auc', 'std_auc', 'max_pct_deviation_auc']])
    print(report_stats[['mean_pr', 'std_pr', 'max_pct_deviation_pr']])
    
    # report.to_word_clipboard()
    
    return summary


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc


def shap_plots():
    


    dataset = pd.read_excel(
        "analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx",
        engine='openpyxl'
    )
    
    cell_line = 'A375'
    context_length = 20
    target = "BE39"
    use_gene_strand = False
    use_efficiency = False
    use_ruleset2 = False
    
    Xf = buildXf(dataset, cell_line=cell_line, use_gene_strand=use_gene_strand, use_efficiency=use_efficiency, use_ruleset2=use_ruleset2)
    Xf = buildXf(dataset, Xf, context=context_length, use_gene_strand=use_gene_strand, use_efficiency=use_efficiency, use_ruleset2=use_ruleset2)
    Xf
    featureset = ['sequence', 'strand', 'depmap', 'domains', 'melt', 'gc', 'mutation_bin']
    if use_ruleset2:
        featureset.append('RuleSet2')
    if use_efficiency:
        featureset.extend(['efficiency'])
    
    X, F, G = buildX(
        Xf,
        context=context_length,
        featureset=featureset
    )
    genes = dataset['Gene symbol'].unique()
    
    

    
    #
    # carico i valori delle shap values
    #   per CatBoost con max_depth=6 e contex_length=20
    
    OFOLDER = "analysis/2024-02-paper1/results/paper1_rev"
    SVs = {}
    for cell_line in cell_lines:
        svs = []
        for gene in tqdm(genes):
            svfile = pathlib.Path(
                f"{OFOLDER}/classifier-cell_line{cell_line}-context_length20-max_depth6-modelCatBoostClassifier-targetBE39-use_efficiencyFalse-use_gene_strandFalse-use_ruleset2False-gene{gene}.SV.joblib")
            
            SV = joblib.load(svfile)
            svs.append(SV)
        SV = np.concatenate(svs)
        SVs[cell_line] = SV
    SVs['A375'].shape[1] == len(F)


    #
    # Figure 4
    #

    df_shap = pd.DataFrame(SVs['A375'], columns=F)

    records = []
    for col in F:
        vals = df_shap[col].values
        records.append({
            'feature': col,
            'importance': np.abs(vals).mean()
        })

    
    df_shapanalysis = pd.DataFrame(records)
    
    # 3. Mappiamo le feature nei gruppi usando il dizionario G (gestendo sia liste che stringhe)
    def get_group(f):
        val = G.get(f, f)
        if isinstance(val, list):
            return val[0]  # Prende il primo tag (es. 'seq:1') o puoi usare un filtro per 'seq'
        return val
    
    df_shapanalysis['feature-group'] = df_shapanalysis['feature'].map(get_group)
    
    # 4. Aggreghiamo per gruppo (sommando le importanze)
    # df_grouped = df_shapanalysis[["feature-group", "importancep", "importancen", "importance"]].groupby('feature-group').sum().reset_index()
    df_grouped = df_shapanalysis[["feature-group", "importance"]].groupby('feature-group').sum().reset_index()
    
    # 1. Isoliamo e ordiniamo le posizioni sequenziali da 1 a 20
    pos_df = df_grouped[df_grouped['feature-group'].str.startswith('seq:')].copy()
    pos_df['pos'] = pos_df['feature-group'].apply(lambda x: int(x.split(':')[1]))
    pos_df.sort_values('pos', inplace=True)
    
    # Rinominiamo i gruppi in pos:1, pos:2, ... pos:20
    pos_df['feature-group'] = [f"pos:{i}" for i in pos_df['pos']]
    pos_df = pos_df[['feature-group', 'importance']]
    
    # 2. Prendiamo tutte le altre feature non sequenziali (es. Mutation, depmap, melt, gc, ecc.)
    other_df = df_grouped[~df_grouped['feature-group'].str.startswith('seq:')]
    
    # 3. Uniamo tutto in un unico DataFrame finale pronto per il plot
    expanded_df = pd.concat([pos_df, other_df], ignore_index=True)
    
    # 6. Ordinamento finale per il plot
    top_features = expanded_df.sort_values("importance", ascending=True)
    top_features.shape
    
    features = top_features["feature-group"].values
    len(features)
    values = top_features["importance"].values
    values

    # rinomino all'ultimo
    features[features == 'strand'] = 'sgRNA strand'
    features[features == 'gc'] = 'GC content'
    features[features == 'melt'] = 'Melting Temperature'

    features
    values
    # 7. Generazione del plot orizzontale
    fig = plt.figure(figsize=(10, 8))
    y = np.arange(len(features))
    
    plt.barh(y, values, color="green", alpha=0.7)
    plt.yticks(y, features)
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Features SHAP values (A375)")
    plt.tight_layout()
    plt.show()
    fig.to_clipboard()
    
    
    #
    # figure 5
    #
    # 1. Dizionario per raccogliere i DataFrame di importanza per ciascuna cell line
    cell_line_dfs = {}
    
    for cell_line in cell_lines:
        # Supponendo che SVs sia il dizionario con i SHAP values per cell line
        df_shap = pd.DataFrame(SVs[cell_line], columns=F)
        
        # Calcolo importanza assoluta media per feature
        records = []
        for col in F:
            vals = df_shap[col].values
            records.append({
                'feature': col,
                'importance': np.abs(vals).mean()
            })
        df_feat = pd.DataFrame(records)
        
        # Mappatura nei gruppi usando G
        df_feat['feature-group'] = df_feat['feature'].map(
            lambda f: G.get(f, f)[0] if isinstance(G.get(f, f), list) else G.get(f, f))
        df_grouped = df_feat.groupby('feature-group')['importance'].sum().reset_index()
        
        # Espansione/gestione delle posizioni sequenziali (seq:1...seq:20 rinominate in pos:1...pos:20)
        pos_df = df_grouped[df_grouped['feature-group'].str.startswith('seq:')].copy()
        if not pos_df.empty:
            pos_df['pos'] = pos_df['feature-group'].apply(lambda x: int(x.split(':')[1]))
            pos_df.sort_values('pos', inplace=True)
            pos_df['feature-group'] = [f"pos:{i}" for i in pos_df['pos']]
            pos_df = pos_df[['feature-group', 'importance']]
            
            other_df = df_grouped[~df_grouped['feature-group'].str.startswith('seq:')]
            expanded_df = pd.concat([pos_df, other_df], ignore_index=True)
        else:
            expanded_df = df_grouped
        
        cell_line_dfs[cell_line] = expanded_df
    
    cell_line_dfs['A375']
    cell_line_dfs['MELJUSO']
    cell_line_dfs['MELJUSO']
    
    # 2. Selezione delle top feature globali (es. unione delle top 10 di ciascuna cell line o tutte)
    topkfeatures = set()
    for cell_line in cell_lines:
        topkfeatures = topkfeatures.union(
            cell_line_dfs[cell_line].sort_values('importance', ascending=False)['feature-group'].head(10)
        )
    
    # 3. Costruzione della tabella pivot per il barplot raggruppato
    bardata = []
    for cell_line in cell_lines:
        bardata.append(
            cell_line_dfs[cell_line].set_index('feature-group')
            .reindex(list(topkfeatures))
            .rename(columns={'importance': cell_line})
        )
    
    
    bardata = pd.concat(bardata, axis=1).fillna(0)
    
    bardata['avg'] = bardata.mean(axis=1)
    bardata.sort_values('avg', ascending=False, inplace=True)
    del bardata['avg']
    # bardata = bardata.head(10)
    features = np.array(bardata.index)
    features[features == 'strand'] = 'sgRNA strand'
    features[features == 'gc'] = 'GC content'
    features[features == 'melt'] = 'Melting Temperature'
    
    bardata.index.name = ''
    # 4. Generazione del grafico a barre verticali raggruppate
    fig, ax = plt.subplots(figsize=(12, 6))
    bardata.plot.bar(stacked=False, ax=ax)
    ax.set_ylabel("Mean absolute SHAP value")
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_title("Top 10 feature SHAP values across cell lines")
    plt.tight_layout()
    plt.show()
    
    fig.to_clipboard()
    
    #
    # figure 5b
    #
    seq_features = [f"pos:{i}" for i in range(1, 21)]
    
    bardata_seq = []
    for cell_line in cell_lines:
        df_seq = cell_line_dfs[cell_line][
            cell_line_dfs[cell_line]["feature-group"].isin(seq_features)
        ]
        bardata_seq.append(
            df_seq.set_index("feature-group")
            .reindex(seq_features)
            .rename(columns={"importance": cell_line})
        )
    
    bardata_seq = pd.concat(bardata_seq, axis=1).fillna(0)
    bardata_seq.index.name = ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bardata_seq.plot.bar(stacked=False, ax=ax)
    ax.set_ylabel("Mean absolute SHAP value")
    ax.set_xticklabels(seq_features, rotation=0, ha="center")
    ax.set_title("Sequence position SHAP values across cell lines")
    plt.tight_layout()
    plt.show()
    
    fig.to_clipboard()

def pi_plots():
    
    dataset = pd.read_excel(
        "analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx",
        engine='openpyxl'
    )

    cell_line = 'A375'
    context_length = 20
    target = "BE39"
    use_gene_strand = False
    use_efficiency = False
    use_ruleset2 = False
    
    Xf = buildXf(dataset, cell_line="A375", use_gene_strand=False, use_efficiency=False, use_ruleset2=False)
    Xf = buildXf(dataset, Xf, context=20, use_gene_strand=False, use_efficiency=False, use_ruleset2=False)
    Xf
    featureset = ['sequence', 'strand', 'depmap', 'domains', 'melt', 'gc', 'mutation_bin']
    if use_ruleset2:
        featureset.append('RuleSet2')
    if use_efficiency:
        featureset.extend(['efficiency'])
    
    X, F, G = buildX(
        Xf,
        context=context_length,
        featureset=featureset
    )
    genes = dataset['Gene symbol'].unique()
    
    Xf['depmap']
    
    OFOLDER = "analysis/2024-02-paper1/results/paper1_rev"
    cell_lines = ('A375', 'MELJUSO', 'OVCAR8', 'HAP1', 'HA1E')  # o le tue cell lines attive
    
    # 1. Funzione di supporto per mappare la singola feature al suo gruppo macro
    def get_group(f):
        val = G.get(f, f)
        if isinstance(val, list):
            if len(val) > 1:
                return val[1]
            elif len(val) == 1:
                return val[0]
            else:
                return "other"
        return val
    
    # Otteniamo la lista ordinata dei gruppi unici basandoci sulle tue feature F
    feature_groups = sorted(list(set(get_group(f) for f in F)))
    
    normalization = 'by-genes'
    # normalization = 'by-replicate'
    
    agg_dict = {}
    for cell_line in cell_lines:
        if normalization == 'by-genes':
            group_vectors_per_gene = []
            for gene in tqdm(genes, desc=f"Processing {cell_line} (by-genes)"):
                pifile = pathlib.Path(
                    f"{OFOLDER}/classifier-cell_line{cell_line}-context_length20-max_depth6-modelCatBoostClassifier-targetBE39-use_efficiencyFalse-use_gene_strandFalse-use_ruleset2False-gene{gene}.PI.joblib"
                )
                data_pi = joblib.load(pifile)
                PI = data_pi['importances_mean']
                i = np.where(F=='depmap')[0][0]
                len(F), F[i]
                print(PI[i])
                
                df_gene_feat = pd.DataFrame({
                    'feature': F,
                    'importance': PI
                })
                df_gene_feat[df_gene_feat['feature'] == 'depmap']
                df_gene_feat['feature-group'] = df_gene_feat['feature'].map(get_group)
                df_grouped_gene = df_gene_feat.groupby('feature-group')['importance'].sum()
                
                gene_vector = [df_grouped_gene.get(g, 0.0) for g in feature_groups]
                group_vectors_per_gene.append(gene_vector)
            
            PI_groups_matrix = np.stack(group_vectors_per_gene)
            n_genes = len(genes)
            
            agg_dict[cell_line] = pd.DataFrame({
                'feature-group': feature_groups,
                'avg': PI_groups_matrix.mean(axis=0),
                'std': PI_groups_matrix.std(axis=0) / np.sqrt(n_genes)
            })
        
        elif normalization == 'by-replicate':
            # Accumuliamo i dataframe di tutte le repliche/geni per mediare direttamente le std interne
            all_replicate_dfs = []
            
            for gene in tqdm(genes, desc=f"Processing {cell_line} (by-replicate)"):
                pifile = pathlib.Path(
                    f"{OFOLDER}/classifier-cell_line{cell_line}-context_length20-max_depth6-modelCatBoostClassifier-targetBE39-use_efficiencyFalse-use_gene_strandFalse-use_ruleset2False-gene{gene}.PI.joblib"
                )
                data_pi = joblib.load(pifile)
                PI_mean = data_pi['importances_mean']
                PI_std = data_pi['importances_std']
                
                df_gene_feat = pd.DataFrame({
                    'feature': F,
                    'mean': PI_mean,
                    'var': PI_std ** 2
                })
                df_gene_feat['feature-group'] = df_gene_feat['feature'].map(get_group)
                
                df_grouped = df_gene_feat.groupby('feature-group').agg({'mean': 'sum', 'var': 'sum'})
                df_grouped['std'] = np.sqrt(df_grouped['var'])
                df_grouped = df_grouped.reset_index()
                all_replicate_dfs.append(df_grouped[['feature-group', 'mean', 'std']])
            
            combined_df = pd.concat(all_replicate_dfs, ignore_index=True)
            agg_dict[cell_line] = (
                combined_df.groupby('feature-group')
                .agg({'mean': 'mean', 'std': 'mean'})
                .rename(columns={'mean': 'avg'})
                .reset_index()
            )
        else:
            raise NotImplementedError(f"Normalization {normalization} not implemented yet")
    
    agg_dict[cell_lines[0]]
    agg_dict[cell_lines[1]]
    agg_dict[cell_lines[2]]
    agg_dict[cell_lines[3]]
    agg_dict[cell_lines[4]]
    
    agg = {
        cell_line: agg_dict[cell_line]
        # .groupby('feature-group')
        # .agg({'avg': 'mean', 'std': 'mean'})
        .reset_index()
        .sort_values('avg', ascending=False)
        for cell_line in cell_lines
    }
    
    #
    # feature_groups = ['seq', 'gc', 'melt', 'Mutation', 'depmap', 'domains', 'strand']
    # aggstats = pd.concat([
    #     agg[cell_line].copy().assign(cell_line=cell_line).set_index('feature-group').loc[feature_groups]
    #     for cell_line in cell_lines
    # ], axis=0).reset_index()
    # aggstats.sort_values('feature-group')
    # aggstats.groupby(['feature-group']).agg(
    #     min_avg=('avg', 'min'),
    #     max_avg=('avg', 'max'),
    #     mean_avg=('avg', 'mean'),
    # ).reset_index()
    #
    # aggstats
    # aggstats[aggstats == 'seq'] = 'Sequence'
    # aggstats[aggstats == 'strand'] = 'sgRNA strand'
    # aggstats[aggstats == 'gc'] = 'GC content'
    # aggstats[aggstats == 'melt'] = 'Melting Temperature'
    #
    
    mapping_labels = {
        'seq': 'Sequence',
        'strand': 'sgRNA strand',
        'gc': 'GC content',
        'melt': 'Melting Temperature',
    }
    
    # 1. Rinominiamo nella lista dei gruppi (usata per l'asse Y)
    feature_groups = [mapping_labels.get(g, g) for g in feature_groups]
    
    # 2. Rinominiamo la colonna 'feature-group' nei DataFrame dentro agg_dict e agg
    for cell_line in cell_lines:
        agg_dict[cell_line]['feature-group'] = agg_dict[cell_line][
            'feature-group'
        ].map(lambda x: mapping_labels.get(x, x))
        agg[cell_line]['feature-group'] = agg[cell_line]['feature-group'].map(
            lambda x: mapping_labels.get(x, x)
        )

    
    fig = plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_groups))
    bar_height = 0.8 / len(cell_lines)
    for i, cell_line in enumerate(cell_lines):
        # data = agg[cell_line].iloc[0:len(features)].iloc[::-1]
        data = agg[cell_line].set_index('feature-group').loc[feature_groups].reset_index().iloc[::-1]
        # data.loc[data['feature']=='behive', 'feature'] = 'BE-Hive'
        plt.barh(
            y_pos + i * bar_height,
            data['avg'],
            xerr=data['std'],
            height=bar_height,
            label=cell_line
        )
    # features[4] = 'BE-Hive'
    
    plt.yticks(y_pos + bar_height * (len(cell_lines) - 1) / 2, feature_groups[::-1])
    xt, xl = plt.xticks()
    plt.xticks(xt, [f"{label * 100:3.1f}" for label in xt])
    plt.xlabel("AUC-ROC drop % (mean ± std)")
    plt.ylabel("Feature group")
    plt.title("Feature importance across cell lines")
    plt.legend(title="Cell line", loc="lower right")
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
    

    fig.to_clipboard()
    
    #
    # Figure 7: Sequence position Permutation Importance across cell lines
    #
    cell_line_pi_seq_dict = {}
    
    for cell_line in cell_lines:
        group_vectors_per_gene_seq = []
        group_vectors_std_per_gene_seq = []
        
        for gene in tqdm(genes, desc=f"Processing {cell_line} seq positions (style)"):
            pifile = pathlib.Path(
                f"{OFOLDER}/classifier-cell_line{cell_line}-context_length20-max_depth6-modelCatBoostClassifier-targetBE39-use_efficiencyFalse-use_gene_strandFalse-use_ruleset2False-gene{gene}.PI.joblib"
            )
            data_pi = joblib.load(pifile)
            PI = data_pi['importances_mean']
            
            df_gene_feat = pd.DataFrame({
                'feature': F,
                'importance': PI
            })
            df_gene_feat['feature-group'] = df_gene_feat['feature'].map(
                lambda f: G.get(f, f)[0] if isinstance(G.get(f, f), list) else G.get(f, f)
            )
            df_grouped_gene = df_gene_feat.groupby('feature-group')['importance'].sum()
            
            seq_features_keys = [f"seq:{i}" for i in range(1, 21)]
            gene_vector = [df_grouped_gene.get(g, 0.0) for g in seq_features_keys]
            group_vectors_per_gene_seq.append(gene_vector)
        
        PI_seq_matrix = np.stack(group_vectors_per_gene_seq)
        n_genes = len(genes)
        
        cell_line_pi_seq_dict[cell_line] = pd.DataFrame({
            'feature': [f"seq:{i}" for i in range(1, 21)],
            'avg': PI_seq_matrix.mean(axis=0),
            'std': PI_seq_matrix.std(axis=0) / np.sqrt(n_genes)
        })
    
    with contextlib.nullcontext():
        features = [f"seq:{i}" for i in range(1, 21)]
        x = np.arange(len(features))
        bar_width = 0.8 / len(cell_lines)
        
        fig = plt.figure(figsize=(10, 6))
        
        for i, cell_line in enumerate(cell_lines):
            data = cell_line_pi_seq_dict[cell_line].set_index('feature').reindex(features)
            
            plt.bar(
                x + i * bar_width,
                data['avg'],
                yerr=data['std'],
                width=bar_width,
                label=cell_line,
                capsize=2
            )
        
        plt.xticks(
            x + bar_width * (len(cell_lines) - 1) / 2,
            [str(i) for i in range(1, 21)],
            rotation=0,
            ha='center'
        )
        
        plt.ylabel("AUC-ROC drop % (mean ± SD)")
        plt.xlabel("Sequence position")
        xt, xl = plt.yticks()
        plt.yticks(xt, [f"{label * 100:3.1f}" for label in xt])
        plt.title("Sequence position Permutation Importance across cell lines")
        plt.legend(title="Cell line")
        plt.tight_layout()
        plt.grid(axis='y')
        plt.show()
    

    fig.to_clipboard()
    
    pass
        
    



def main():
    training_set = pd.read_excel("analysis/2024-02-paper1/dataset/hanna/dataset-efficiency-depmap-domains.xlsx",
                                 engine='openpyxl')
    validation_set = pd.read_excel("analysis/2024-02-paper1/dataset/coelho/dataset-efficiency-depmap-domains.xlsx",
                                   engine='openpyxl')
    
    training_set[['BE39:A375:zscore', 'CAS9:A375:zscore']]
    
    cell_lines = 'A375', 'MELJUSO', 'OVCAR8', 'HAP1', 'HA1E'
    (training_set[[f'BE39:{cell_line}:zscore' for cell_line in cell_lines]].values < -2).mean(0)
    
    for context_length in [20, ]:
        for cell_line in cell_lines:
        # for cell_line in ('OVCAR8', 'HAP1', 'HA1E'):
        #     print(cell_line, context_length)
            
            train(
                training_set=training_set,
                model='GaussianNB',
                # model='LGBMClassifier',
                # model='XGBClassifier',
                # model='CatBoostClassifier',
                # model='RandomForest',
                cell_line=cell_line,
                context_length=context_length,
                target="BE39",  # or CAS9
                use_gene_strand=False,
                force_training=False,
                do_shap=True,
                do_perm=True,
            
            )
    
    Y, Spred = validate(
        # validation_set="analysis/2024-02-paper1/dataset/coelho/dataset-efficiency-depmap-domains.xlsx",
        validation_set=validation_set,
        model='RandomForestClassifier',
        # model='LGBMClassifier',
        # model='GaussianNB',
        cell_line='A375',
        # cell_line='MELJUSO',
        # cell_line="OVCAR8",
        # cell_line="HAP1",
        # cell_line="HA1E",
        context_length=20,
        
        target="BE39",  # or CAS9
        use_gene_strand=False,
        zscore_threshold=-2
    )
    
    print(roc_auc_score(Y, Spred))
    balanced_accuracy_score(Y, Spred.round())
    
    sel_gene = (validation_set['Gene symbol'] == 'JAK1').values
    print(roc_auc_score(Y[sel_gene], Spred[sel_gene]))
    accuracy_score(Y[sel_gene], Spred[sel_gene].round())
    balanced_accuracy_score(Y[sel_gene], Spred[sel_gene].round())
    
    sel_gene = (validation_set['Gene symbol'] != 'JAK1').values
    print(roc_auc_score(Y[sel_gene], Spred[sel_gene]))
    accuracy_score(Y[sel_gene], Spred[sel_gene].round())
    balanced_accuracy_score(Y[sel_gene], Spred[sel_gene].round())
    
    # validation_set.groupby("Gene symbol")
    validation_set['Gene symbol'].value_counts()
    
    # validation_set["sgRNA_ID"].str.startswith("CTRL", na=False)
    
    # calcolo le accuracy per ogni gene
    validation_set.groupby('Gene symbol')
    
    gene_accuracy_report = (
        validation_set.assign(Ytrue=Y, Ypred=Spred.round())
        .groupby("Gene symbol")
        .apply(
            lambda group: pd.Series(
                {
                    "Guide_Count": len(group),
                    "Accuracy": balanced_accuracy_score(group["Ytrue"], group["Ypred"]),
                }
            ),
            include_groups=False,
        )
        .sort_values(by="Accuracy", ascending=False)
    )
    gene_accuracy_report.to_clipboard()
    
    print(roc_auc_score(Y, Spred))


if __name__ == '__main__':
    main()
    # result_plots()
