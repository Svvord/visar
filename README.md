
# VISAR Tutorial

This project aims to train neural networks by compound-protein interactions and provides interpretation of the learned model by interactively showing transformed chemical landscape and visualized SAR for chemicals of interest.

In this notebook, we will show a typical workflow of using VISAR for training neural network QSAR models and analyzing the trained model.

## model training
With VISAR, we could set the hyperparameters candidate sets for screening, and manually pick the best one.
The training is set to carried out 3 times as repeats, and training results at various learning steps are saved in log directory for further analysis.


```python
import os
from Model_training_utils import ST_model_hyperparam_screen, ST_model_training
os.environ['CUDA_VISIBLE_DEVICES']='1'
```


```python
# initialize parameters
task_names = ['T51'] # see dataset table in data directory for all available tasks
MT_dat_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'

params_dict = {
    "n_tasks": [1],
    "n_features": [2048], ## need modification given FP types
    "activation": ['relu'],
    "momentum": [.9],
    "batch_size": [128],
    "init": ['glorot_uniform'],
    "learning_rate": [0.01],
    "decay": [1e-6],
    "nb_epoch": [30],
    "dropouts": [.2, .4],
    "nb_layers": [1],
    "batchnorm": [False],
    #"layer_sizes": [(100, 20), (64, 24)],
    "layer_sizes": [(1024, 512),(1024,128) ,(512, 128),(512,64),(128,64),(64,32), 
                    (1024,512,128), (512,128,64), (128,64,32)],
    "penalty": [0.1]
}
```


```python
# initialize model setup
import random
import time
random_seed = random.randint(0,1000)
local_time = time.localtime(time.time())
log_path = './logs/'
RUN_KEY = 'ST_%d_%d_%d_%d' % (local_time.tm_year, local_time.tm_mon, 
                              local_time.tm_mday, random_seed)
os.system('mkdir %s%s' % (log_path, RUN_KEY))
print(RUN_KEY)
```


```python
# hyperparam screening using deepchem
RUN_KEY = 'ST_2019_4_23_136'
log_output = ST_model_hyperparam_screen(MT_dat_name, task_names, FP_type, params_dict, 
                                        log_path = './logs/'+RUN_KEY)
```


```python
# manually pick the training parameters
best_hyperparams = {'T51': [(512, 128,1), 0.2]
                   }
```


```python
# model training
RUN_KEY = 'ST_2019_4_23_136'
output_df = ST_model_training(MT_dat_name, FP_type, 
                              best_hyperparams, result_path = './logs/'+RUN_KEY)
```

## build landscape and display interactive plot
Once specified the task name and the name of the trained model, the 'landscape_building' function would carry out the analysis, and the result could be interactively displayed by the function 'interactive_plot'.


```python
from Model_landscape_utils import landscape_building, interactive_plot
from Model_training_utils import prepare_dataset, extract_clean_dataset
from keras import backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
from bokeh.plotting import output_notebook, show
output_notebook()
```

    Using TensorFlow backend.
    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="ffb4a7ea-b82e-4f57-8174-d5cb03eebed1">Loading BokehJS ...</span>
    </div>





```python
# analysis set-up
task_name = 'T51'
db_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'
log_path = './logs/ST_2019_4_23_136/'
prev_model = './logs/ST_2019_4_23_136/T51_rep2_50.hdf5'
n_layer = 1
SAR_result_dir = log_path
output_sdf_name = log_path + 'T51_chemical_landscape.sdf'
```


```python
plot_df = landscape_building(task_name, db_name, log_path, FP_type,
                       prev_model, n_layer, 
                       SAR_result_dir, output_sdf_name)
plot_df.head()
```

    ==== preparing dataset ... ====
    Extracted dataset shape: (3202, 3)
    Loading raw samples now.
    shard_size: 8192
    About to start loading CSV from ./logs/ST_2019_4_23_136//temp.csv
    Loading shard 1 of size 8192.
    Featurizing sample 0
    Featurizing sample 1000
    Featurizing sample 2000
    Featurizing sample 3000
    TIMING: featurizing shard 0 took 14.154 s
    TIMING: dataset construction took 14.381 s
    Loading dataset from disk.
    ==== calculating transfer values ... ====
    WARNING:tensorflow:From /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:1108: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    ==== rendering SAR for chemicals on the landscape ... ====
    ==== packing sdf file ... ====
    

    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:410: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if np.issubdtype(type(cell_value), float):
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T51</th>
      <th>molregno</th>
      <th>salt_removed_smi</th>
      <th>coord1</th>
      <th>coord2</th>
      <th>Label</th>
      <th>imgs</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>1.346777</td>
      <td>262348</td>
      <td>Cc1nn2c(C#N)cccc2c1CN1CCN(c2ccc(Cl)cc2)CC1</td>
      <td>-61.390247</td>
      <td>28.508858</td>
      <td>78</td>
      <td>./logs/ST_2019_4_23_136/262348_img.png</td>
      <td>1.431429</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1.657567</td>
      <td>262350</td>
      <td>N#Cc1cccc2c(CN3CCN(c4ccc(Cl)cc4)CC3)cnn12</td>
      <td>-57.633320</td>
      <td>30.381008</td>
      <td>99</td>
      <td>./logs/ST_2019_4_23_136/262350_img.png</td>
      <td>1.712855</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.999990</td>
      <td>322</td>
      <td>COc1ccc2c(c1)[nH]c1c(C)nccc12</td>
      <td>-62.837112</td>
      <td>18.570761</td>
      <td>63</td>
      <td>./logs/ST_2019_4_23_136/322_img.png</td>
      <td>1.038729</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.999990</td>
      <td>351</td>
      <td>COc1ccc2c3c([nH]c2c1)CNCC3</td>
      <td>-66.213913</td>
      <td>19.925594</td>
      <td>102</td>
      <td>./logs/ST_2019_4_23_136/351_img.png</td>
      <td>1.037158</td>
    </tr>
    <tr>
      <th>104</th>
      <td>4.892833</td>
      <td>262631</td>
      <td>N[C@@H](Cc1c[nH]c2ccc(O)cc12)C(=O)O</td>
      <td>38.830685</td>
      <td>-5.267498</td>
      <td>9</td>
      <td>./logs/ST_2019_4_23_136/262631_img.png</td>
      <td>4.937616</td>
    </tr>
  </tbody>
</table>




```python
# make interactive plot
show(interactive_plot(plot_df, x_column = 'coord1', y_column = 'coord2', color_column = task_name, 
                      id_field = 'molregno', value_field = task_name, label_field = 'Label', 
                      pred_field = 'pred', size = 6))
```








  <div class="bk-root" id="e4d007f3-0991-4d1a-994a-db1ad6da6f15"></div>






```python
# a helping function of loading dataframe from previously generated sdf file
from Model_landscape_utils import sdf2df
plot_df = sdf2df('./logs/ST_2019_4_23_136/T51_chemical_landscape.sdf')
plot_df['Label'] = [int(x) for x in plot_df['Label'].tolist()]
```


```python
plot_df.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>molregno</th>
      <th>imgs</th>
      <th>salt_removed_smi</th>
      <th>coord1</th>
      <th>coord2</th>
      <th>T51</th>
      <th>pred</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>262348</td>
      <td>./logs/ST_2019_4_23_136/262348_img.png</td>
      <td>Cc1nn2c(C#N)cccc2c1CN1CCN(c2ccc(Cl)cc2)CC1</td>
      <td>59.324326</td>
      <td>45.583145</td>
      <td>1.346777</td>
      <td>1.431429</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>262350</td>
      <td>./logs/ST_2019_4_23_136/262350_img.png</td>
      <td>N#Cc1cccc2c(CN3CCN(c4ccc(Cl)cc4)CC3)cnn12</td>
      <td>59.331421</td>
      <td>40.36301</td>
      <td>1.657567</td>
      <td>1.712855</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>322</td>
      <td>./logs/ST_2019_4_23_136/322_img.png</td>
      <td>COc1ccc2c(c1)[nH]c1c(C)nccc12</td>
      <td>51.826027</td>
      <td>50.381916</td>
      <td>0.99999</td>
      <td>1.038729</td>
      <td>88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>351</td>
      <td>./logs/ST_2019_4_23_136/351_img.png</td>
      <td>COc1ccc2c3c([nH]c2c1)CNCC3</td>
      <td>43.928574</td>
      <td>52.7785</td>
      <td>0.99999</td>
      <td>1.037158</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>262631</td>
      <td>./logs/ST_2019_4_23_136/262631_img.png</td>
      <td>N[C@@H](Cc1c[nH]c2ccc(O)cc12)C(=O)O</td>
      <td>-34.64777</td>
      <td>13.794934</td>
      <td>4.892833</td>
      <td>4.937616</td>
      <td>78</td>
    </tr>
  </tbody>
</table>




```python
# pick clusters of interest and pack them as an sdf for pharmacophore modeling
from Model_landscape_utils import df2sdf
import numpy as np
output_sdf_name = log_path + 'T51_Label13_chemical_landscape.sdf'
smiles_field = 'salt_removed_smi'
id_field = 'molregno'
filter_label = np.array([x == 21 for x in plot_df['Label'].tolist()])
custom_df = plot_df.loc[filter_label]
df2sdf(custom_df, output_sdf_name, smiles_field, id_field)
```

    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:296: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      frame[molCol] = frame[smilesCol].map(Chem.MolFromSmiles)
    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:410: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if np.issubdtype(type(cell_value), float):
    

## Pharmacophore modeling using Align-it


```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
raw_sdf_file = './T51_Label13_chemical_landscape.sdf'
ms = [x for x in Chem.SDMolSupplier(raw_sdf_file)]
len(ms)
```


```python
# set home directory for following analysis
home_dir = './Result/'
os.chdir(home_dir)

# prepare ligand conformations
from rdkit import Chem
from rdkit.Chem import AllChem

raw_sdf_file = 'Label_7.sdf'
sdf_file = home_dir + 'Label7_rdkit_conf.sdf'
ms = [x for x in Chem.SDMolSupplier(raw_sdf_file)]
n_conf = 5
w = Chem.SDWriter(sdf_file)
for i in range(n_conf):
    ms_addH = [Chem.AddHs(m) for m in ms]
    for m in ms_addH:
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMoleculeConfs(m)
        w.write(m)

# process pharmacophores
result_dir = home_dir + 'Label7_rdkit_phars/'
output_name = 'Cluster7_'
proceed_pharmacophore(home_dir, sdf_file, result_dir, output_name)
```

The resulting pharamacophore model (saved as .phar files at home_dir) could be visualized in Pymol by align-it plugin.

## analysis of custom chemicals
For new set of chemical of interst, users could prepare a seperate .csv file, with columns specifying the SMILES, ID and a dummy field of biological activity, and the 'landscape_positioning' function would take in the custom file and generate analysis results.


```python
from Model_landscape_utils import landscape_positioning, interactive_plot
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
from bokeh.plotting import output_notebook, show
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="75c98913-bdfd-4a0d-a58c-a8aeeeee0c08">Loading BokehJS ...</span>
    </div>





```python
# set custom file
custom_file = './Result/custom_df.csv'
custom_smi_field = "smiles"
custom_id_field = 'molname'
custom_task_field = 'dummy'

# set the landscape to compare to
landscape_sdf = './logs/ST_2019_4_23_136/T51_chemical_landscape.sdf'
task_name = 'T51'
db_name = './data/MT_data_clean_Feb28.csv'
FP_type = 'Circular_2048'
log_path = './logs/'
prev_model = './logs/ST_2019_4_23_136/T51_rep2_50.hdf5'
n_layer = 1
custom_SAR_result_dir = log_path
custom_sdf_name = log_path + 'custom_chemicals_on_T51_landscape.sdf'
```


```python
plot_df = landscape_positioning(custom_file, custom_smi_field, custom_id_field, custom_task_field,
                        landscape_sdf, task_name, db_name, FP_type, log_path,
                        prev_model, n_layer, custom_SAR_result_dir, custom_sdf_name)
```

    ==== preparing dataset ... ====
    Extracted dataset shape: (3202, 3)
    Loading raw samples now.
    shard_size: 8192
    About to start loading CSV from ./logs//temp.csv
    Loading shard 1 of size 8192.
    Featurizing sample 0
    Featurizing sample 1000
    Featurizing sample 2000
    Featurizing sample 3000
    TIMING: featurizing shard 0 took 14.038 s
    TIMING: dataset construction took 14.269 s
    Loading dataset from disk.
    Extracted dataset shape: (2, 3)
    Loading raw samples now.
    shard_size: 8192
    About to start loading CSV from ./Result/custom_df.csv
    Loading shard 1 of size 8192.
    Featurizing sample 0
    TIMING: featurizing shard 0 took 0.011 s
    TIMING: dataset construction took 0.021 s
    Loading dataset from disk.
    ==== calculating transfer values ... ====
    ==== rendering SAR for chemicals on the landscape ... ====
    ==== packing sdf file ... ====
    

    /root/anaconda3/envs/deepchem/lib/python3.5/site-packages/rdkit/Chem/PandasTools.py:410: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if np.issubdtype(type(cell_value), float):
    


```python
plot_df.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coord1</th>
      <th>coord2</th>
      <th>molregno</th>
      <th>T51</th>
      <th>Label</th>
      <th>salt_removed_smi</th>
      <th>imgs</th>
      <th>pred</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.507763</td>
      <td>1.298865</td>
      <td>RIT</td>
      <td>0.000000</td>
      <td>65</td>
      <td>CC1=C(CCN2CCC(=C(c3ccc(F)cc3)c4ccc(F)cc4)CC2)C(=O)N5C=CSC5=N1</td>
      <td>./logs/RIT_img.png</td>
      <td>3.344871</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17.061337</td>
      <td>53.975483</td>
      <td>ERG</td>
      <td>0.000000</td>
      <td>59</td>
      <td>CN1C[C@@H](C=C2[C@H]1Cc3c[nH]c4cccc2c34)C(=O)N[C@]5(C)O[C@@]6(O)[C@@H]7CCCN7C(=O)[C@H](Cc8ccccc8)N6C5=O</td>
      <td>./logs/ERG_img.png</td>
      <td>4.034642</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-38.608498</td>
      <td>-23.622177</td>
      <td>262348</td>
      <td>1.346777</td>
      <td>104</td>
      <td>Cc1nn2c(C#N)cccc2c1CN1CCN(c2ccc(Cl)cc2)CC1</td>
      <td>./logs/ST_2019_4_23_136/262348_img.png</td>
      <td>2.174304</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-47.691116</td>
      <td>-16.576090</td>
      <td>262350</td>
      <td>1.657567</td>
      <td>58</td>
      <td>N#Cc1cccc2c(CN3CCN(c4ccc(Cl)cc4)CC3)cnn12</td>
      <td>./logs/ST_2019_4_23_136/262350_img.png</td>
      <td>2.069756</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-90.054886</td>
      <td>-12.611736</td>
      <td>322</td>
      <td>0.999990</td>
      <td>13</td>
      <td>COc1ccc2c(c1)[nH]c1c(C)nccc12</td>
      <td>./logs/ST_2019_4_23_136/322_img.png</td>
      <td>1.043407</td>
      <td>6</td>
    </tr>
  </tbody>
</table>




```python
show(interactive_plot(plot_df, 'coord1', 'coord2', color_column = 'group', id_field = 'molregno', 
                      value_field = task_name, label_field = 'Label',pred_field = 'pred', size = 'group'))
```








  <div class="bk-root" id="b9219084-8f2c-4e0b-b836-c5eb19b5d517"></div>






```python
from Model_landscape_utils import sdf2df
plot_df = sdf2df('./logs/custom_chemicals_on_T51_landscape.sdf')
plot_df['Label'] = [int(x) for x in plot_df['Label'].tolist()]
```


```python
plot_df.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>molregno</th>
      <th>T51</th>
      <th>imgs</th>
      <th>Label</th>
      <th>coord1</th>
      <th>coord2</th>
      <th>pred</th>
      <th>salt_removed_smi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RIT</td>
      <td>0.0</td>
      <td>./logs/RIT_img.png</td>
      <td>65</td>
      <td>5.507763</td>
      <td>1.298865</td>
      <td>3.344871</td>
      <td>CC1=C(CCN2CCC(=C(c3ccc(F)cc3)c4ccc(F)cc4)CC2)C(=O)N5C=CSC5=N1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ERG</td>
      <td>0.0</td>
      <td>./logs/ERG_img.png</td>
      <td>59</td>
      <td>17.061337</td>
      <td>53.975483</td>
      <td>4.034642</td>
      <td>CN1C[C@@H](C=C2[C@H]1Cc3c[nH]c4cccc2c34)C(=O)N[C@]5(C)O[C@@]6(O)[C@@H]7CCCN7C(=O)[C@H](Cc8ccccc8)N6C5=O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>262348</td>
      <td>1.346777</td>
      <td>./logs/ST_2019_4_23_136/262348_img.png</td>
      <td>104</td>
      <td>-38.608498</td>
      <td>-23.622177</td>
      <td>2.174304</td>
      <td>Cc1nn2c(C#N)cccc2c1CN1CCN(c2ccc(Cl)cc2)CC1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>262350</td>
      <td>1.657567</td>
      <td>./logs/ST_2019_4_23_136/262350_img.png</td>
      <td>58</td>
      <td>-47.691116</td>
      <td>-16.57609</td>
      <td>2.069756</td>
      <td>N#Cc1cccc2c(CN3CCN(c4ccc(Cl)cc4)CC3)cnn12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>322</td>
      <td>0.99999</td>
      <td>./logs/ST_2019_4_23_136/322_img.png</td>
      <td>13</td>
      <td>-90.054886</td>
      <td>-12.611736</td>
      <td>1.043407</td>
      <td>COc1ccc2c(c1)[nH]c1c(C)nccc12</td>
    </tr>
  </tbody>
</table>




```python
# pick clusters of interest and pack them as an sdf fur pharmacophore modeling
custom_filter = landscape_df['Label'] == 7
df2sdf(df, output_sdf_name, smiles_field, id_field, custom_filter = None)
```
