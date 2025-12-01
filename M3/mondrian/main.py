import argparse
from mondrian import mondrian
from utils.data_loader import load_data, build_categorical_hierarchy, build_range_hierarchy
from utils import display_table, default_data_config


def check_uniqueness_risk(df, quasi_id, sensitive):
    group_sizes = df.groupby(quasi_id).size()
    unique_groups = (group_sizes == 1).sum()
    sensitive_leakage = df.groupby(quasi_id)[sensitive].nunique()
    single_sensitive = (sensitive_leakage == 1).sum()
    return unique_groups, single_sensitive


def main(config):
    data = load_data(config=config)
    table = data['table']

    before_unique, before_sensitive_leak = check_uniqueness_risk(table, data['quasi_id'], data['sensitive'])

    encoders = {}
    from utils.data_loader import preprocess_categorical_column, recover_categorical_mondrian
    for attr in data['quasi_id']:
        if config['data']['mondrian_generalization_type'][attr] == 'categorical':
            table[attr], encoder = preprocess_categorical_column(table[attr].tolist())
            encoders[attr] = encoder

    anonymized_table, loss_metric = mondrian(
        table=table,
        quasi_id=data['quasi_id'],
        k=config['k'],
        sensitive=config['data']['sensitive']
    )

    for attr in data['quasi_id']:
        if config['data']['mondrian_generalization_type'][attr] == 'categorical':
            table[attr] = recover_categorical_mondrian(table[attr].tolist(), encoders[attr])

    anonymized_table = anonymized_table[data['quasi_id'] + [data['sensitive']]]
    display_table(anonymized_table)
    anonymized_table.to_csv('results/mondrian.csv', header=None, index=None)

    after_unique, after_sensitive_leak = check_uniqueness_risk(anonymized_table, data['quasi_id'], data['sensitive'])

    print(f"\n[Uniqueness] Before: {before_unique} | After: {after_unique}")
    print(f"[Sensitive Leakage Risk] Before: {before_sensitive_leak} | After: {after_sensitive_leak}")

    return loss_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=10, type=int)
    config = vars(parser.parse_args())
    config['data'] = default_data_config
    print('\nconfiguration:\n', config)

    main(config)
