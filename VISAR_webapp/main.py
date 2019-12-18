import os
import pandas as pd
import numpy as np
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering

from bokeh.palettes import Category20_20, Category20b_20, BuPu,RdBu11
from bokeh.plotting import figure
from bokeh.models import Panel, Tabs
from bokeh.models.widgets import (
    Select, 
    Slider, 
    TextInput, 
    PreText, 
    Button, 
    TextInput,
    RadioButtonGroup
)

import matplotlib.cm as cm
import matplotlib as mpl
from bokeh.io import show, output_notebook
from bokeh.models import (
    ColumnDataSource, 
    Div,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    FactorRange
)

from bokeh.layouts import row, column
from bokeh.io import curdoc

from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
from random import sample

from SAR_plot_utils import plot_SAR

#####----------------------------------
#------------------------------------------------
# functions for processing input data into plots
def update_grid_chemical(batch_selected = None):
    global click_cnt, compound_df, custom_df

    if batch_selected is None:
        sample_index = sample(range(len(compound_df)), 15)
    else:
        index_selected = compound_df.index[compound_df['label'] == batch_selected]
        if len(index_selected) <= 15:
            sample_index = index_selected
        else:
            sample_index = sample(list(index_selected), 15)

    mols = [Chem.MolFromSmiles(compound_df['canonical_smiles'].iloc[idx]) for idx in sample_index]
    
    cnt = 0
    for mol in mols:
        mol.SetProp('__Name', str(compound_df['chembl_id'].iloc[sample_index[cnt]]))
        cnt += 1
    for mol in mols: tmp=AllChem.Compute2DCoords(mol)
    img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(125,90),legends=[x.GetProp("__Name") for x in mols])
    img.save('./VISAR_webapp/static/foo_%d.png' % click_cnt)

    foo_img = Div(text="<img src='VISAR_webapp/static/foo_%d.png', height='300', width='650'>" % click_cnt)
    click_cnt += 1
    return foo_img

def update_SAR_chemical(task_id ,chem_idx = 0):
    global SAR_cnt, compound_df, custom_df, task_df, MODE
    if radio_button_group.active == 0:
        plot_SAR(compound_df, task_df, chem_idx, task_id, SAR_cnt, mode = MODE, cutoff = 10, n_features = 2048)
    elif radio_button_group.active == 1:
        plot_SAR(custom_df, task_df, chem_idx, task_id, SAR_cnt, mode = MODE, cutoff = 10, n_features = 2048)
    sar_img = Div(text="<img src='VISAR_webapp/static/SAR_%d.png', height='200', width='300'>" % SAR_cnt)
    SAR_cnt += 1
    return sar_img

def update_bicluster(K = None):
    global batch_df, task_df, compound_df, custom_df, MODE, CUSTOM_FLAG

    if MODE == 'RobustMT':
        n_tasks = task_df.shape[1] - 1
    elif MODE == 'ST' or MODE == 'baseline':
        n_tasks = 1
    elif MODE == 'MT':
        n_tasks = task_df.shape[1]
    elif MODE == 'AttentiveFP':
        n_tasks = sum([item[0:4] == 'avg_' for item in batch_df.columns.values])
    
    if MODE == 'RobustMT' or MODE == 'AttentiveFP':
        # cocluster of the minibatch predictive matrix
        X = preprocessing.scale(np.matrix(batch_df)[:,0:n_tasks])
        cocluster = SpectralCoclustering(n_clusters=K, random_state=0)
        cocluster.fit(X)
        batch_df['batch_label'] = cocluster.row_labels_
    elif MODE == 'ST' or MODE == 'baseline':
        rank_x = batch_df[batch_df.columns[0]].rank().tolist()
        groups = pd.qcut(rank_x, K, duplicates='drop')
        batch_df['batch_label'] = groups.codes
    
    # generate color hex for batch_label
    lut = dict(zip(batch_df['batch_label'].unique(), Category20_20))
    batch_df['batch_label_color'] = batch_df['batch_label'].map(lut)
    
    # generate color hex for compound_df
    lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
    compound_df['batch_label_color'] = compound_df['label'].map(lut2)
    lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
    compound_df['batch_label'] = compound_df['label'].map(lut22)
    groups = pd.qcut(compound_df['label'].tolist(), len(Category20b_20), duplicates='drop')
    c = [Category20b_20[xx] for xx in groups.codes]
    compound_df['label_color'] = c

    #--------- generate heatmap dataframe -------
    if MODE == 'RobustMT' or MODE == 'AttentiveFP':
        X = preprocessing.scale(np.matrix(batch_df)[:,0:n_tasks])
        fit_data = X[np.argsort(cocluster.row_labels_)]
        fit_data = fit_data[:, np.argsort(cocluster.column_labels_)].T
        [min_value, max_value] = [fit_data.min().min(), fit_data.max().max()]
    
        # prepare dataframe structure for bokeh plot 
        if MODE == 'RobustMT':
            rows = task_df.columns[0:-1]
        else:
            rows = [item[4:] for item in batch_df.columns.values if item[0:4] == 'avg_']
        rows_new = [rows[idx] for idx in np.argsort(cocluster.column_labels_)]
        cols = batch_df['Label_id'].tolist()
        cols_new = [str(cols[idx]) for idx in np.argsort(cocluster.row_labels_)]

        plot_dat = pd.DataFrame(fit_data)
        plot_dat['task'] = rows_new
        plot_dat = plot_dat.set_index('task')
        plot_dat.columns = cols_new
        plot_dat.columns.name = 'label'

    
    elif MODE == 'ST' or MODE == 'baseline':
        X = np.asarray(np.matrix(batch_df)[:,0:n_tasks]).reshape(-1)
        order_x = np.asarray(np.argsort(X)).reshape(-1)
        fit_data = X[order_x].T
        [min_value, max_value] = [fit_data.min(), fit_data.max()]
        
        rows_new = [DEFAULT_TICKERS[2]]
        cols = batch_df['Label_id'].tolist()
        cols_new = [str(cols[idx]) for idx in order_x]

        plot_dat = pd.DataFrame(np.asmatrix(fit_data))
        plot_dat['task'] = rows_new
        plot_dat = plot_dat.set_index('task')
        plot_dat.columns = cols_new
        plot_dat.columns.name = 'label'

    plot_df = pd.DataFrame(plot_dat.stack(), columns = ['value']).reset_index()
    lut = dict(zip(batch_df['Label_id'].astype('str'), batch_df['batch_label_color']))
    plot_df['batch_label_color'] = plot_df['label'].map(lut)

    if CUSTOM_FLAG:
        lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
        lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
        lut222 = dict(zip(compound_df['label'], compound_df['label_color']))
        custom_df['batch_label_color'] = custom_df['label'].map(lut2)
        custom_df['batch_label'] = custom_df['label'].map(lut22)
        custom_df['label_color'] = custom_df['label'].map(lut222)
    
    return plot_df, compound_df, custom_df, [min_value, max_value], [cols_new, rows_new]

# set up plots
def set_up_heatmap(plot_df, value_range, RC_names):
    global N_TASK

    source.data = plot_df[['task','label','value','batch_label_color']] 
    TOOLTIPS = [
        ("index", "$index"),
        ("batch id", "@label"),
        ("cluster color", "$color[hex, swatch]:batch_label_color")
    ]
    colormap =cm.get_cmap("BuPu")
    bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

    mapper = LinearColorMapper(palette=bokehpalette,low=value_range[0], high=value_range[1])
    z = figure(x_range=RC_names[0], y_range=RC_names[1], toolbar_location='below', 
               plot_width = 650, plot_height = (N_TASK * 10 + 30), toolbar_sticky=False, 
               tools = 'pan,box_zoom,tap,reset,save,hover', tooltips=TOOLTIPS)
    z.rect(x='label', y='task', width=1, height=1, source=source, 
           fill_color={'field': 'value', 'transform': mapper}, line_color=None,
           nonselection_fill_alpha = 0.2)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=8),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    z.add_layout(color_bar, 'right')
    z.xaxis.axis_label = 'label'
    z.yaxis.axis_label = 'task'
    z.xaxis.visible = False

    return z

def set_up_scatter(plot_df2, value_field = 'batch_label'):
    global compound_df_columns, custom_compound_df_columns, DEFAULT_TICKERS, DEFAULT_CUSTOM_TICKERS
    global x_min, x_max, y_min, y_max

    TOOLTIPS = [
        ("index", "$index"),
        ("compound id", "@chembl_id"),
        ("batch id", "@label"),
        ("cluster color", "$color[hex, swatch]:batch_label_color")
    ]

    p = figure(x_range=[x_min - 5, x_max + 5], 
               y_range=[y_min - 5, y_max + 5],
               plot_width = 650, plot_height = 500, toolbar_location = 'right',
               tools = 'pan,box_zoom,reset,lasso_select,tap,save,hover', tooltips=TOOLTIPS)

    # color rendering
    if radio_button_group.active == 0:
        if value_field == 'batch_label':
            color_field = 'batch_label_color'
            cds_df.data = plot_df2[compound_df_columns]

        elif value_field == 'label':
            color_field = 'label_color'
            cds_df.data = plot_df2[compound_df_columns]

        elif value_field in DEFAULT_TICKERS:
            COLORS = RdBu11
            N_COLORS = len(COLORS)
            plot_df2 = plot_df2.loc[~np.isnan(plot_df2[value_field])]
            groups = pd.qcut(plot_df2[value_field].tolist(), N_COLORS, duplicates='drop')
            c = [COLORS[xx] for xx in groups.codes]
            plot_df2['c'] = c
            color_field = 'c'
            new_columns = compound_df_columns + ['c']
            cds_df.data = plot_df2[new_columns]
        else:
            print('Ilegal value field')
            return

    if radio_button_group.active == 1:
        if value_field == 'batch_label':
            color_field = 'batch_label_color'
            cds_df.data = plot_df2[custom_compound_df_columns]

        elif value_field == 'label':
            color_field = 'label_color'
            cds_df.data = plot_df2[custom_compound_df_columns]

        elif value_field in DEFAULT_CUSTOM_TICKERS and len(plot_df2[value_field].unique()) <= 20:
            plot_df2 = plot_df2.loc[~pd.isnull(plot_df2[value_field])]
            lut = dict(zip(plot_df2[value_field].unique(), Category20_20))
            plot_df2['c'] = plot_df2[value_field].map(lut)
            color_field = 'c'
            new_columns = custom_compound_df_columns + ['c']
            cds_df.data = plot_df2[new_columns]

        elif value_field in DEFAULT_CUSTOM_TICKERS:
            COLORS = RdBu11
            N_COLORS = len(COLORS)
            plot_df2 = plot_df2.loc[~pd.isnull(plot_df2[value_field])]
            groups = pd.qcut(plot_df2[value_field].tolist(), N_COLORS, duplicates = 'drop')
            c = [COLORS[xx] for xx in groups.codes]
            plot_df2['c'] = c
            color_field = 'c'
            new_columns = custom_compound_df_columns + ['c']
            cds_df.data = plot_df2[new_columns]

        else:
            print('Ilegal value field')
            return
    
    p.circle(source = cds_df, x = 'x', y = 'y', color=color_field, size=5,
            nonselection_fill_alpha=0.2, nonselection_fill_color = 'grey')

    return p

#-------------------------
# set up callbacks
def update_heatmap(attr, old, new):
    plot_df, plot_df2, plot_df3, value_range, RC_names = update_bicluster(K = input_k.value)
    layout.children[1].children[0].children[2] = set_up_heatmap(plot_df, value_range, RC_names)
    p_training = set_up_scatter(plot_df2, value_field = select.value)
    layout.children[1].children[0].children[1] = p_training

def update_landscape(attr, old, new):
    _, plot_df2, _, _, _ = update_bicluster(K = input_k.value)
    if radio_button_group.active == 1:
        layout.children[1].children[0].children[1] = set_up_scatter(plot_df3, value_field = select.value)
    elif radio_button_group.active == 0:
        layout.children[1].children[0].children[1] = set_up_scatter(plot_df2, value_field = select.value)

def update_cluster_img(attr, old, new):
    global compound_df, custom_df

    batch_selected = int(select_cluster.value)
    update_stats(source)
    plot_df, plot_df2, plot_df3, _, _ = update_bicluster(K = input_k.value)
    if batch_selected and radio_button_group.active == 0:
        source.selected.indices = [plot_df.index[plot_df['label'] == str(batch_selected)].tolist()[0]]
        cds_df.selected.indices = list(compound_df.index[compound_df['label'] == batch_selected])
    if batch_selected and radio_button_group.active == 1:
        source.selected.indices = [plot_df.index[plot_df['label'] == str(batch_selected)].tolist()[0]]
        cds_df.selected.indices = list(custom_df.index[custom_df['label'] == batch_selected])


def update_SAR1(attrname, old, new):
    selected = cds_df.selected.indices
    if selected:
        chem_idx = list(selected)[0]
        layout.children[1].children[1].children[3].children[0] = update_SAR_chemical(task_select1.value, chem_idx)

def update_SAR2(attrname, old, new):
    selected = cds_df.selected.indices
    if selected:
        chem_idx = list(selected)[0]
        layout.children[1].children[1].children[3].children[1] = update_SAR_chemical(task_select2.value, chem_idx)

def update_stats(source):
    global batch_df, batch_info_columns
    selected = source.selected.indices
    if selected:
        batch_selected = int(source.data['label'][selected[0]])
        selected_batch_data = batch_df[batch_df['Label_id'] == batch_selected]
        stats.text = str(selected_batch_data[batch_info_columns])

def update_radio_button(attrname, old, new):
    global plot_df2, plot_df3, compound_df_columns, custom_compound_df_columns
    if radio_button_group.active == 1:
        select.options = DEFAULT_CUSTOM_TICKERS
        select.value = 'batch_label'
        cds_df.data = plot_df3[custom_compound_df_columns]
        layout.children[1].children[0].children[1] = set_up_scatter(plot_df3, value_field = select.value)
    elif radio_button_group.active == 0:
        select.options = DEFAULT_TICKERS
        select.value = 'batch_label'
        cds_df.data = plot_df2[compound_df_columns]
        layout.children[1].children[0].children[1] = set_up_scatter(plot_df2, value_field = select.value)

def selection_change(attrname, old, new):
    update_stats(source)
    selected = source.selected.indices
    _, plot_df2, plot_df3, _, _ = update_bicluster(K = input_k.value)
    if selected:
        batch_selected = int(source.data['label'][selected[0]])
        layout.children[1].children[1].children[1] = update_grid_chemical(batch_selected = batch_selected)
        if radio_button_group.active == 0:
            cds_df.selected.indices = list(compound_df.index[compound_df['label'] == batch_selected])
            layout.children[1].children[0].children[1] = set_up_scatter(plot_df2, value_field = select.value)
        elif radio_button_group.active == 1:
            cds_df.selected.indices = list(custom_df.index[custom_df['label'] == batch_selected])
            layout.children[1].children[0].children[1] = set_up_scatter(plot_df3, value_field = select.value)

def sar_selection_change(attrname, old, new):
    selected = cds_df.selected.indices
    if selected:
        chem_idx = list(selected)[0]
        layout.children[1].children[1].children[3].children[0] = update_SAR_chemical(task_select1.value, chem_idx)
        layout.children[1].children[1].children[3].children[1] = update_SAR_chemical(task_select2.value, chem_idx)

#--------------------------------------------------------
# update the whole dataset
def update_database(attrname):
    global compound_df, custom_df, batch_df, task_df, MODE, N_TASK, CUSTOM_FLAG
    global DEFAULT_TICKERS, batch_info_columns, compound_df_columns
    global DEFAULT_CUSTOM_TICKERS, custom_compound_df_columns
    global x_min, x_max, y_min, y_max, plot_df2, plot_df3

    # load new datasets
    DATA_DIR = file_prefix_input.value
    compound_df = pd.read_csv(DATA_DIR + 'compound_df.csv', )
    batch_df = pd.read_csv(DATA_DIR + 'batch_df.csv')
    task_df = pd.read_csv(DATA_DIR + 'task_df.csv')
    MODE = mode_select.value

    CUSTOM_FLAG = False
    if os.path.isfile(DATA_DIR + 'compound_custom_df.csv'):
        CUSTOM_FLAG = True
        custom_df = pd.read_csv(DATA_DIR + 'compound_custom_df.csv')
    else:
        custom_df = compound_df

    [x_min, x_max, y_min, y_max] = [min(min(compound_df['x']), min(custom_df['x'])),
                                    max(max(compound_df['x']), max(custom_df['x'])),
                                    min(min(compound_df['y']), min(custom_df['y'])),
                                    max(max(compound_df['y']), max(custom_df['y']))]

    # setup global parameters
    if MODE == 'RobustMT':
        N_TASK = task_df.shape[1] - 1
    elif MODE == 'ST' or MODE == 'baseline':
        N_TASK = 1
    elif MODE == 'AttentiveFP':
        N_TASK = sum([item[0:4] == 'avg_' for item in batch_df.columns.values])

    batch_info_columns = ['Label_id', 'size'] + list(batch_df.columns)[0:N_TASK] 
    compound_df_columns = list(compound_df.columns)
    custom_compound_df_columns = list(custom_df.columns)
    cds_dict = {}
    for x in compound_df_columns:
        cds_dict[x] = []
    cds_dict['c'] = []
    cds_df.data = cds_dict

    DEFAULT_TICKERS = ['label', 'batch_label'] + \
        [x[5:] for x in list(compound_df.columns) if x[0:4] == 'pred'] + \
        [x for x in list(compound_df.columns) if x[0:4] == 'pred']
    DEFAULT_CUSTOM_TICKERS = [x for x in custom_compound_df_columns 
        if x not in ['CID', 'SMILES', 'canonical_smiles', 'x', 'y',
                     'chembl_id', 'cid', 
                     'label_color','batch_label_color']]

    if MODE == 'RobustMT':
        DEFAULT_TASKS = list(task_df.columns)
    elif MODE == 'ST' or MODE == 'baseline':
        DEFAULT_TASKS = [DEFAULT_TICKERS[2]]
    elif MODE == 'AttentiveFP':
        DEFAULT_TASKS = ['SHARE']

    select.options = DEFAULT_TICKERS
    select.value = 'batch_label'
    radio_button_group.active = 0

    DEFAULT_batches = [str(xx) for xx in batch_df['Label_id'].tolist()]    
    select_cluster.value = DEFAULT_batches[0]
    select_cluster.options = DEFAULT_batches

    task_select1.value = DEFAULT_TASKS[-1]
    task_select2.value = DEFAULT_TASKS[0]
    task_select1.options = DEFAULT_TASKS
    task_select2.options = DEFAULT_TASKS

    # setup figures and plots again
    plot_df, plot_df2, plot_df3, value_range, RC_names = update_bicluster(K = input_k.value)

    layout.children[1].children[0].children[1] = set_up_scatter(plot_df2, value_field = select.value)
    layout.children[1].children[0].children[2] = set_up_heatmap(plot_df, value_range, RC_names)

    layout.children[1].children[1].children[1] = update_grid_chemical(batch_selected = None)
    layout.children[1].children[1].children[3].children[0] = update_SAR_chemical(task_id = task_select1.value)
    layout.children[1].children[1].children[3].children[1] = update_SAR_chemical(task_id = task_select2.value)
    
    return

#####-----------------------------------


## setup visualization mode and data source
file_prefix_input = TextInput(value='VISAR_webapp/data/RobustMT_Kinase_K12_', title='Prefix of the input data:')
mode_select = Select(title='Mode of the model', value='RobustMT', 
                     options=['RobustMT', 'MT', 'ST', 'baseline', 'AttentiveFP'])

DATA_DIR = file_prefix_input.value
compound_df = pd.read_csv(DATA_DIR + 'compound_df.csv', )
batch_df = pd.read_csv(DATA_DIR + 'batch_df.csv')
task_df = pd.read_csv(DATA_DIR + 'task_df.csv')
CUSTOM_FLAG = False
if os.path.isfile(DATA_DIR + 'compound_custom_df.csv'):
    CUSTOM_FLAG = True
    custom_df = pd.read_csv(DATA_DIR + 'compound_custom_df.csv')
else:
    custom_df = compound_df
MODE = mode_select.value
[x_min, x_max, y_min, y_max] = [min(min(compound_df['x']), min(custom_df['x'])),
                                max(max(compound_df['x']), max(custom_df['x'])),
                                min(min(compound_df['y']), min(custom_df['y'])),
                                max(max(compound_df['y']), max(custom_df['y']))]

if MODE == 'RobustMT':
    N_TASK = task_df.shape[1] - 1
elif MODE == 'ST' or MODE == 'baseline':
    N_TASK = 1
elif MODE == 'AttentiveFP':
    N_TASK = sum([item[0:4] == 'avg_' for item in batch_df.columns.values])

batch_info_columns = ['Label_id', 'size'] + list(batch_df.columns)[0:N_TASK] 

## prepare bokeh data source for webapp
compound_df_columns = list(compound_df.columns)
cds_dict = {}
for x in compound_df_columns:
    cds_dict[x] = []
cds_dict['c'] = []

custom_compound_df_columns = list(custom_df.columns)

cds_df = ColumnDataSource(data = cds_dict)
source = ColumnDataSource(data=dict(task=[], label=[], value=[], batch_label_color=[]))

DEFAULT_TICKERS = ['label', 'batch_label'] + \
        [x[5:] for x in list(compound_df.columns) if x[0:4] == 'pred'] + \
        [x for x in list(compound_df.columns) if x[0:4] == 'pred']
DEFAULT_CUSTOM_TICKERS = [x for x in custom_compound_df_columns 
        if x not in ['CID', 'SMILES', 'canonical_smiles', 'x', 'y',
                     'chembl_id', 'cid',
                     'label_color','batch_label_color']]

if MODE == 'RobustMT':
    DEFAULT_TASKS = list(task_df.columns)
elif MODE == 'ST' or MODE == 'baseline':
    DEFAULT_TASKS = [DEFAULT_TICKERS[2]]
elif MODE == 'AttentiveFP':
    DEFAULT_TASKS = ['SHARE']

#-----------------------------
## set up widgets
select = Select(title='Color options for chemical landscape', value='batch_label', options=DEFAULT_TICKERS)
select.on_change('value', update_landscape)

input_k = Slider(title="Number of clusters", value=5, start=3, end=20, step=1)
input_k.on_change('value', update_heatmap)

DEFAULT_batches = [str(xx) for xx in batch_df['Label_id'].tolist()]
select_cluster = Select(title='Select batch id', value = DEFAULT_batches[0], options = DEFAULT_batches)
select_cluster.on_change('value', update_cluster_img)

task_select1 = Select(title='SAR of task ...', value=DEFAULT_TASKS[-1], options = DEFAULT_TASKS)
task_select1.on_change('value', update_SAR1)

task_select2 = Select(title='SAR of task ...', value=DEFAULT_TASKS[0], options = DEFAULT_TASKS)
task_select2.on_change('value', update_SAR2)

stats = PreText(text = '', width = 650)

source.selected.on_change('indices', selection_change)
cds_df.selected.on_change('indices', sar_selection_change)

run_button = Button(label='RUN', button_type='success')
run_button.on_click(update_database)

radio_button_group = RadioButtonGroup(labels = ['training data', 'custom data'], active = 0)
radio_button_group.on_change('active', update_radio_button)

#--------------------------------------------------------
# set up image
click_cnt = 0
SAR_cnt = 0
foo_img = update_grid_chemical(batch_selected = None)
SAR_img1 = update_SAR_chemical(task_id = task_select1.value)
SAR_img2 = update_SAR_chemical(task_id = task_select2.value)

# set up layout
plot_df, plot_df2, plot_df3, value_range, RC_names = update_bicluster(K = input_k.value)
p_training = set_up_scatter(plot_df2)

layout = column(
    row(file_prefix_input, mode_select, run_button,radio_button_group),
    row(column(row(select,input_k), 
               p_training,
               set_up_heatmap(plot_df, value_range, RC_names),
               stats), 
        column(select_cluster, 
               foo_img, 
               row(task_select1,task_select2), 
               row(SAR_img1, SAR_img2)))
    )

# initialize
curdoc().add_root(layout)
curdoc().title = "VISAR"


# possible race condition: https://discourse.bokeh.org/t/possible-race-condition-when-replacing-figure-object/951/3

