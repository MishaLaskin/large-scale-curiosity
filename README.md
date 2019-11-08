**Status:** Archive (code is provided as-is, no updates expected)

## Large-Scale Study of Curiosity-Driven Learning ##
#### [[Project Website]](https://pathak22.github.io/large-scale-curiosity/) [[Demo Video]](https://youtu.be/l1FqtAHfJLI)

[Yuri Burda*](https://sites.google.com/site/yburda/), [Harri Edwards*](https://github.com/harri-edwards/), [Deepak Pathak*](https://people.eecs.berkeley.edu/~pathak/), <br/>[Amos Storkey](http://homepages.inf.ed.ac.uk/amos/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)<br/>
(&#42; alphabetical ordering, equal contribution)

University of California, Berkeley<br/>
OpenAI<br/>
University of Edinburgh

<a href="https://pathak22.github.io/large-scale-curiosity/">
<img src="https://pathak22.github.io/large-scale-curiosity/resources/teaser.jpg" width="500">
</img></a>

This is a TensorFlow based implementation for our [paper on large-scale study of curiosity-driven learning](https://pathak22.github.io/large-scale-curiosity/) across
54 environments. Curiosity is a type of intrinsic reward function which uses prediction error as reward signal. In this paper, We perform the first large-scale study of purely curiosity-driven learning, i.e. without any extrinsic rewards, across 54 standard benchmark environments. We further investigate the effect of using different feature spaces for computing prediction error and show that random features are sufficient for many popular RL game benchmarks, but learned features appear to generalize better (e.g. to novel game levels in Super Mario Bros.). If you find this work useful in your research, please cite:

    @inproceedings{largeScaleCuriosity2018,
        Author = {Burda, Yuri and Edwards, Harri and
                  Pathak, Deepak and Storkey, Amos and
                  Darrell, Trevor and Efros, Alexei A.},
        Title = {Large-Scale Study of Curiosity-Driven Learning},
        Booktitle = {arXiv:1808.04355},
        Year = {2018}
    }

### Installation and Usage
The following command should train a pure exploration agent on Breakout with default experiment parameters.
```bash
python run.py
```
To use more than one gpu/machine, use MPI (e.g. `mpiexec -n 8 python run.py` should use 1024 parallel environments to collect experience instead of the default 128 on an 8 gpu machine).

### Saving Agents

The policy is automatically saved during training. Specify a path to your desired folder with the `--saved_model_dir` flag, e.g.:

```
python run.py --saved_model_dir YOUR_SAVED_MODEL_FOLDER
```

The model filename is hardcoded to be `model.ckpt`, but in your folder you'll find `model.ckpt-{INTEGER}` files where integer corresponds to a specific save since the model is saved periodically.

### Restoring Saved Agents

Once you've ran and saved an agent, you can restore it by specifying the saved model folder and the model name with the `--saved_model_dir` and `--model_name` flags. The model name should be `model.ckpt-{INTEGER}` without the `.index` suffix, so `model.ckpt-100` would be a valid model name.

### Evaluating a Restored Agent

In order to evaluate you have to enable the `-eval` flag. You can specify the evaluation length with the `--n_eval_steps` argument, which is set to 512 steps by default. Note: the default number of workers is 128 workers and is set by `--envs_per_process`, so if you run the default configuration you will get 128 parallel rollouts of 512 length each.  

Running evaluation will automatically generate a dataset of saved actions. You can also get image observations from the evaluation run if you enable the `-include_images` flag. If you include images, the data file size can get quite large (~1.8G for 128 workers x 512 steps per worker).

Here's an example command for running evaluation and collecting a dataset that includes images:

```
python run.py -eval -include_images --saved_model_dir ./tmp/ --model_name model.ckpt-100
```

### Visualizing the Evaluation Dataset 

Now that you've saved your data, you should have a file names `{env_name}_data.npy` (for example, `BreakoutNoFrameskip-v4_data.npy`) in your current directory. To visualize, open the `Visualization.ipynb` notebook and run it. You should get a video of the saved rollouts if you included images. Sense check these to make sure the curiosity code is working correctly.

### Data for plots in paper

[Data for Figure-2](https://www.dropbox.com/s/ufr7o8g9omb9zpl/experiments.tar.gz): contains raw game score data along with the plotting code to generate Figure-2 in the paper.


### Other helpful pointers
- [Paper](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf)
- [Project Website](https://pathak22.github.io/large-scale-curiosity/)
- [Demo Video](https://youtu.be/l1FqtAHfJLI)
