from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def line(df, parameter, make_virtual_best=False, save_dir=''):
    try:
        # Remove unknowns and errors
        df = df[(df['satisfied'] != 'unknown') & (df['satisfied'] != 'error')]

        if make_virtual_best:
            virt_best = df.groupby(['model', 'query'], group_keys=False).min()
            virt_best['experiment'] = 'VirtualBest'
            virt_best.reset_index(inplace=True)
            df = pd.concat([df, virt_best])

        # Rank rows
        df = df.groupby('experiment', group_keys=True).apply(lambda x: x.sort_values(parameter).reset_index(drop=True))
        df = df.reset_index(level=1, names=[None, 'rank']).reset_index(drop=True)

        # Display
        sns.set_theme(style='whitegrid')
        ax = sns.relplot(data=df, kind='line', x='rank', y=parameter, hue='experiment')
        ax.set(yscale='log')

        if save_dir != '':
            # Save
            png = graphs_dir / save_dir / f'{parameter}.png'
            png.parent.mkdir(exist_ok=True)
            plt.savefig(png)
            print(f'Created {save_dir}/{parameter}.png')

            eps = graphs_dir / save_dir / f'{parameter}.eps'
            eps.parent.mkdir(exist_ok=True)
            plt.savefig(eps)
            print(f'Created {save_dir}/{parameter}.eps')

            # Interactive version
            fig = px.line(df, x='rank', y=parameter, color='experiment', log_y=True, hover_name='model', hover_data=['formula', 'satisfied'])
            fig.write_html(graphs_dir / save_dir / f'{parameter}.html')
        else:
            print(f'Showing {parameter}.png')
            plt.show()

    # except Exception as err:
    #     print(f'ERR: Failed to line graph showing \'{parameter}\' for \'{save_dir}\'')
    finally:
        plt.close()


def line_by_termination(df, parameter, make_virtual_best=False, save_dir=''):
    try:

        # Remove unknowns and errors
        df = df[(df['satisfied'] != 'unknown') & (df['satisfied'] != 'error')]

        if make_virtual_best:
            virt_best = df.groupby(['model', 'query'], group_keys=False).min()
            virt_best['experiment'] = 'VirtualBest'
            virt_best.reset_index(inplace=True)
            df = pd.concat([df, virt_best])

        # Split by early termination
        df['early term'] = df['formula'].str.contains('A[]', regex=False) == df['satisfied'].str.contains('false', regex=False)
        # df['experiment'] = df['experiment'] + df.T.apply(lambda row: (' ⊤', ' ÷')[row['early term']])

        # Rank rows
        df = df.groupby(['experiment', 'early term'], group_keys=True).apply(lambda x: x.sort_values(parameter).reset_index(drop=True))
        df = df.reset_index(level=2, names=[None, None, 'rank']).reset_index(drop=True)

        # Display
        sns.set_theme(style='whitegrid')
        ax = sns.relplot(data=df, kind='line', x='rank', y=parameter, hue='experiment')
        ax.set(yscale='log')

        if save_dir != '':
            # Save
            png = graphs_dir / save_dir / f'{parameter}.png'
            png.parent.mkdir(exist_ok=True)
            plt.savefig(png)
            print(f'Created {save_dir}/{parameter}.png')

            # Interactive version
            fig = px.line(df, x='rank', y=parameter, color='experiment', line_dash='early term', log_y=True, hover_name='model', hover_data=['formula', 'satisfied'])
            fig.write_html(graphs_dir / save_dir / f'{parameter} by termination.html')
        else:
            print(f'Showing {parameter}.png')
            plt.show()

    # except Exception as err:
    #     print(f'ERR: Failed to line graph showing \'{parameter}\' for \'{save_dir}\'')
    finally:
        plt.close()


def cactus(df, parameter, min_threshold=0, denominator='LU', save_dir=''):
    try:

        den_rows = df['experiment'] == denominator
        den = df[den_rows]
        den = den.set_index(['model', 'query'])[parameter]
        df = df[-den_rows]

        # Avoid division by 0
        df = df[df.T.apply(lambda row: den[row['model'], row['query']] != 0)]

        # Exclude small values of parameter
        df = df[df.T.apply(lambda row: den[row['model'], row['query']] >= min_threshold)]

        # Remove unknowns and errors
        df = df[(df['satisfied'] != 'unknown') & (df['satisfied'] != 'error')]

        # Calculate fraction
        frac_name = f'{parameter}, ratio wrt M' if parameter != 'stored' else f'zones, ratio wrt M'
        df['base'] = df.T.apply(lambda row: den[row['model'], row['query']])
        df[frac_name] = df[parameter] / df['base']
        # df['experiment'] = df['experiment'] + f'/{denominator}'

        # Rank rows
        rank_str = 'models, ranked by ratio'
        df = df.groupby('experiment', group_keys=True).apply(lambda x: x.sort_values(frac_name).reset_index(drop=True))
        df = df.reset_index(level=1, names=[None, rank_str]).reset_index(drop=True)
        df[rank_str] += 1

        # Display
        sns.set_theme(style='whitegrid')
        n_color = len(df['experiment'].unique())
        # Horizontal at 1
        plt.plot([0, df[rank_str].max()], [1,1], color='black', linewidth=0.7)
        ax = sns.lineplot(data=df, x=rank_str, y=frac_name, hue='experiment',
                         palette=sns.color_palette()[1:n_color + 1])
        # ax.set(yscale='log')
        ax.set_ylim(0, None)
        ax.set_xlim(1, df[rank_str].max())
        sns.set(rc={'figure.figsize': (3.7, 2.5)})
        plt.legend(title=None, loc='lower right')
        plt.tight_layout()

        # Save
        if save_dir != '':
            png = graphs_dir / save_dir / f'cactus {parameter}.png'
            png.parent.mkdir(exist_ok=True)
            plt.savefig(png)
            print(f'Created {save_dir}/cactus {parameter}.png')

            eps = graphs_dir / save_dir / f'cactus {parameter}.eps'
            eps.parent.mkdir(exist_ok=True)
            plt.savefig(eps)
            print(f'Created {save_dir}/cactus {parameter}.eps')

            # Interactive version
            fig = px.line(df, x=rank_str, y=frac_name, color='experiment', hover_name='model', hover_data=['formula', 'satisfied'])
            fig.add_hline(y=1, layer='below')
            fig.write_html(graphs_dir / save_dir / f'cactus {parameter}.html')
        else:
            print(f'Showing cactus {parameter}.png')
            plt.show()

        df.to_csv(graphs_dir / save_dir / f'cactus {parameter}.csv', index=False)

    # except KeyError as err:
    #     print(f'ERR: Failed to cactus graph showing \'{parameter}\' for \'{save_dir}\'')
    finally:
        plt.close()


def diagonal(df, parameter, denominator='LU', save_dir=''):

    base_name = f'{parameter} ({denominator})'

    den_rows = df['experiment'] == denominator
    den = df[den_rows].copy()
    den = den.set_index(['model', 'query'])[parameter]
    df = df[-den_rows].copy()

    # Avoid division by 0  # TODO Discard 0s
    # df = df[df.T.apply(lambda row: den[row['model'], row['query']] != 0)]

    # Find base size
    df[base_name] = df.T.apply(lambda row: den[row['model'], row['query']])

    # Remove unknowns and errors
    df = df[(df['satisfied'] != 'unknown') & (df['satisfied'] != 'error')]

    # Diagonal
    baseline = pd.DataFrame({
        base_name: [0, df[parameter].max()],
        parameter: [0, df[parameter].max()],
    })

    # Display
    sns.set_theme(style='whitegrid')
    n_color = len(df['experiment'].unique())
    ax = sns.scatterplot(data=df, x=base_name, y=parameter, hue='experiment', style='experiment',
                         palette=sns.color_palette()[1:n_color + 1])
    ax = sns.lineplot(data=baseline, x=base_name, y=parameter, linewidth=0.7, ax=ax)
    ax.set(xscale='log', yscale='log', ylabel=f'{parameter}')

    # Save
    if save_dir != '':
        png = graphs_dir / save_dir / f'diagonal {parameter}.png'
        png.parent.mkdir(exist_ok=True)
        plt.savefig(png)
        print(f'Created {save_dir}/diagonal {parameter}.png')

        eps = graphs_dir / save_dir / f'diagonal {parameter}.eps'
        eps.parent.mkdir(exist_ok=True)
        plt.savefig(eps)
        print(f'Created {save_dir}/diagonal {parameter}.eps')

        # Interactive version
        fig = px.scatter(df, x=base_name, y=parameter, color='experiment', symbol='experiment', log_x=True, log_y=True, hover_name='model', hover_data=['formula', 'satisfied'])
        fig.update_layout(shapes=[{'type': 'line', 'xref': 'x', 'yref': 'y', 'y0': 1, 'y1': df[base_name].max(), 'x0': 1, 'x1': df[base_name].max(), 'layer': 'below'}])
        fig.write_html(graphs_dir / save_dir / f'diagonal {parameter}.html')
    else:
        print(f'Showing diagonal {parameter}.png')
        plt.show()

    plt.close()


def uniques(df, save_dir=''):

    # Remove unknowns and errors
    df = df[(df['satisfied'] != 'unknown') & (df['satisfied'] != 'error')]

    # Remove instances with more than one answering experiment
    df = df.groupby(['model', 'query']).filter(lambda g: len(g) == 1)

    if save_dir != '':
        df.to_csv(graphs_dir / save_dir / f'uniques.csv', index=False)
        print(f'Created {save_dir}/uniques.csv')


def uniques_wrt_base(df, denominator='LU', save_dir=''):

    # Remove unknowns and errors
    df = df[(df['satisfied'] != 'unknown') & (df['satisfied'] != 'error')]

    # Remove instances with more than one answering experiment
    df = df.groupby(['model', 'query']).filter(lambda g: not g['experiment'].eq(denominator).any())

    if save_dir != '':
        df.to_csv(graphs_dir / save_dir / f'uniques wrt {denominator}.csv', index=False)
        print(f'Created {save_dir}/uniques wrt {denominator}.csv')


def csv_table(df, denominator='LU', save_dir=''):
    df = df[['stored', 'time', 'experiment', 'model']] \
        .pivot(index='model', columns='experiment', values=['stored', 'time'])
    df.columns = ['{} {}'.format(x[1], x[0]) for x in df.columns]

    for col in list(df.columns):
        ex, par = col.split(' ')
        if ex != denominator:
            df[f'{ex}/{denominator} {par}%'] = 100 * df[col] / df[f'{denominator} {par}']

    df = df[sorted(df.columns, key=lambda c: int('/' in c) * 1000 + len(c.split(' ')[0].split('/')[0]) * 100 - len(c.split(' ')[1]))]
    df = pd.concat([df, df.agg(['mean', 'sum'])])
    df = pd.concat([df, pd.DataFrame(
        {col: None if '/' in col else df.loc['mean', col] / df.loc['mean', f'{denominator} {col.split(" ")[1]}']
         for col in df.columns}, index=[f'% wrt {denominator}']
    )])

    df.to_csv(graphs_dir / save_dir / 'table.csv')
    print(f'Created {save_dir}/table.csv')
    df.to_latex(graphs_dir / save_dir / 'table.tex')
    print(f'Created {save_dir}/table.tex')


@dataclass
class DataSet:
    name: str
    csvs: List[Path]
    denominator: str
    virtual_best: bool = True
    do_line_by_termination: bool = False


if __name__ == '__main__':
    file = Path(__file__)
    graphs_dir = file.parent.parent / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    result_dir = file.parent.parent / 'results'

    data_sets = [
        DataSet('with-type-ranges', [
            result_dir / '230424-with-type-ranges' / 'M.csv',
            result_dir / '230424-with-type-ranges' / 'LU.csv',
            result_dir / '230424-with-type-ranges' / 'DynM.csv',
            result_dir / '230424-with-type-ranges' / 'DynLU.csv',
        ], denominator='M', virtual_best=False),
        DataSet('without-type-ranges', [
            result_dir / '230512-without-type-ranges' / 'M.csv',
            result_dir / '230512-without-type-ranges' / 'LU.csv',
            result_dir / '230512-without-type-ranges' / 'DynM.csv',
            result_dir / '230512-without-type-ranges' / 'DynLU.csv',
        ], denominator='M', virtual_best=False),
        DataSet('selected-for-paper', [
            result_dir / '2305xx-selected-for-paper' / 'M.csv',
            result_dir / '2305xx-selected-for-paper' / 'DynM.csv',
        ], denominator='M', virtual_best=False),
    ]

    for ds in data_sets:

        print(f'Processing data set: {ds.name}...')

        # Combine data
        df_all = []
        for path in ds.csvs:
            df_tmp = pd.read_csv(path, sep=';')
            df_tmp['experiment'] = path.stem
            df_all.append(df_tmp)
        df = pd.concat(df_all, ignore_index=True)

        # Graphs
        line(df.copy(deep=True), 'time', make_virtual_best=ds.virtual_best, save_dir=ds.name)
        line(df.copy(deep=True), 'explored', make_virtual_best=ds.virtual_best, save_dir=ds.name)
        line(df.copy(deep=True), 'stored', make_virtual_best=ds.virtual_best, save_dir=ds.name)
        # line(df.copy(deep=True), 'res mem'excluded_formulae=gr.excluded_formualae, make_virtual_best=gr.virtual_best, save_dir=gr.name)
        if ds.do_line_by_termination:
            line_by_termination(df.copy(deep=True), 'time', make_virtual_best=ds.virtual_best, save_dir=ds.name)
            line_by_termination(df.copy(deep=True), 'explored', make_virtual_best=ds.virtual_best, save_dir=ds.name)
        cactus(df.copy(deep=True), 'time', min_threshold=50, denominator=ds.denominator, save_dir=ds.name)
        cactus(df.copy(deep=True), 'explored', denominator=ds.denominator, save_dir=ds.name)
        cactus(df.copy(deep=True), 'stored', denominator=ds.denominator, save_dir=ds.name)
        # cactus(df.copy(deep=True), 'res mem', denominator=gr.denominator, save_dir=gr.name)
        diagonal(df.copy(deep=True), 'time', denominator=ds.denominator, save_dir=ds.name)
        diagonal(df.copy(deep=True), 'explored', denominator=ds.denominator, save_dir=ds.name)
        diagonal(df.copy(deep=True), 'stored', denominator=ds.denominator, save_dir=ds.name)

        uniques(df.copy(deep=True), save_dir=ds.name)
        uniques_wrt_base(df.copy(deep=True), denominator=ds.denominator, save_dir=ds.name)

        csv_table(df.copy(deep=True), denominator=ds.denominator, save_dir=ds.name)

    print(f'Done. Graphs and tables can be found in {graphs_dir}')
