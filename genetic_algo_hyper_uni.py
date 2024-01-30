import data_loading_util
import numpy as np
# Data preparation steps from the provided notebook cells
window_size = 200 
num_snapshots_in_sequence = 256 # How many windows in a single batch
sequence_len = num_snapshots_in_sequence + window_size - 1 # How much data in a batch
num_columns_in_training_tensor = 12 # Left and right input channels + output labels

predict_future_by = 0 # How many samples to predict into the future

subjects_to_train_with = [1,2,3,4,5,6,7,8,9]
sides = ['LEFT', 'RIGHT']

trial_nums = 1+np.arange(11)

root_folder = "complete_labeled_dataset"

training_instances = np.empty(shape=[0,sequence_len, num_columns_in_training_tensor], dtype=np.float32)
training_files = data_loading_util.get_files_to_use(root_folder, subjects_to_train_with, sides, trial_nums)

# Processing data
for file in training_files:
    data = data_loading_util.load_file(file)

    # Process data
    stance_phase_column = -4
    data = data_loading_util.replace_column_values_simple(data, stance_phase_column, 0, -1)
    
    num_rows, num_cols = data.shape
    num_rows_to_drop = num_rows % sequence_len
    data = data[0:-num_rows_to_drop]
    new_num_rows, num_cols = data.shape
    num_sequences = new_num_rows / sequence_len
    new_data_shape = (int(num_sequences), sequence_len, num_cols)
    new_instances = data.reshape(new_data_shape)
    
    training_instances = np.append(training_instances, new_instances, axis=0)

# Additional data processing steps
shuffled_data, x, y_v, y_r, y_sp, y_ss = data_loading_util.shuffle_and_extract_features_labels_uni(training_instances, window_size)
labels = [y_sp, y_ss, y_r, y_v]
x_train, x_valid, labels_train, labels_valid = data_loading_util.split_data(x, labels)
# Extract individual labels
y_sp_train, y_ss_train, y_r_train, y_v_train = labels_train
y_sp_valid, y_ss_valid, y_r_valid, y_v_valid = labels_valid

import tensorflow as tf

def construct_model_2023(window_size,
                         filter_sizes,
                         kernel_sizes,
                         dilations,
                         num_channels=8,
                         batch_norm_insertion_pts=[2],
                         sp_dense_sizes=[20, 10],
                         ss_dense_sizes=[20, 10],
                         v_dense_sizes=[20, 10],
                         r_dense_sizes=[20, 10],

                         do_fix_input_dim=False):
  if len(filter_sizes) != len(kernel_sizes)+1:
      raise ValueError(
          'Must provide one more filter size than kernel size--last kernel size is calculated')
  current_output_size = window_size  # Track for final conv layer

  #Use None in dim 0 to allow variable input length.
  #Use window_size to fix size--helpful for debugging dimensions
  if do_fix_input_dim:
    input_layer = tf.keras.layers.Input(
        shape=(window_size, num_channels), name='my_input_layer')
  else:
    input_layer = tf.keras.layers.Input(
        shape=(None, num_channels), name='my_input_layer')

  z = input_layer
  for layer_idx in range(len(kernel_sizes)):
      z = tf.keras.layers.Conv1D(filters=filter_sizes[layer_idx], kernel_size=kernel_sizes[layer_idx],
                                  dilation_rate=dilations[layer_idx], activation='relu')(z)
      if layer_idx in batch_norm_insertion_pts:
          z = tf.keras.layers.BatchNormalization()(z)
      current_output_size = current_output_size - \
          dilations[layer_idx]*kernel_sizes[layer_idx] + dilations[layer_idx]
  if current_output_size < 1:
      raise ValueError('layers shrink the cnn too much')
  else:
      print('adding final conv layer of kernel size: ', current_output_size)
      last_conv_layer = tf.keras.layers.Conv1D(
          filters=filter_sizes[-1], kernel_size=current_output_size, activation='relu')(z)

  # gait phase DNN
  z = last_conv_layer
  for num_neurons in sp_dense_sizes:
      z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
  output_stance_phase = tf.keras.layers.Dense(1, name='stance_phase_output')(z)

  z = last_conv_layer
  for num_neurons in ss_dense_sizes:
      z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
  output_stance_swing = tf.keras.layers.Dense(
      1, activation='sigmoid', name='stance_swing_output')(z)
  
  z = last_conv_layer
  for num_neurons in v_dense_sizes:
      z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
  velocity = tf.keras.layers.Dense(1, name='velocity_output')(z)

  z = last_conv_layer
  for num_neurons in r_dense_sizes:
      z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
  ramp = tf.keras.layers.Dense(1, name='ramp_output')(z)

  model = tf.keras.Model(inputs=[input_layer], outputs=[
      output_stance_phase, output_stance_swing, velocity, ramp])

  return model

import random
import tensorflow as tf

from keras.callbacks import Callback
from tqdm import tqdm

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_progress = tqdm(total=self.params['steps'],
                                   desc=f'Epoch {self.current_epoch}/{self.epochs}',
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                                   leave=False)  # Set leave=False so progress bar disappears after epoch

    def on_batch_end(self, batch, logs=None):
        # Update the description with the current losses
        desc = f'Epoch {self.current_epoch}/{self.epochs} - loss: {logs["loss"]:.4f}'
        for output in ['stance_phase_output_ipsi', 'stance_swing_output_ipsi', 'velocity_output', 
                       'ramp_output', 'stance_phase_output_contra', 'stance_swing_output_contra']:
            if f'{output}_loss' in logs:
                desc += f' - {output}_loss: {logs[f"{output}_loss"]:.4f}'

        self.epoch_progress.set_description(desc)
        self.epoch_progress.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress.close()

import itertools

# Define the hyperparameter space
hyperparam_space = {
    # Three filter sizes each ranging from 20 to 40
    "filter_sizes": list(itertools.combinations(range(20, 41), 3)),
    
    # Two kernel sizes each centered around 30 (from 25 to 35)
    "kernel_sizes": list(itertools.combinations(range(25, 36), 2)),
    
    # Two dilations each ranging from 1 to 3
    "dilations": list(itertools.product(range(1, 4), repeat=2)),
    
    # Fixed number of channels
    "num_channels": [8],

    # Two batch norm insertion points each ranging from 0 to 3
    "batch_norm_insertion_pts": list(itertools.combinations(range(0, 4), 2)),

    # Three values for each dense size parameter, ranging from 15 to 45
    "sp_dense_sizes": list(itertools.combinations(range(15, 46), 3)),
    "ss_dense_sizes": list(itertools.combinations(range(15, 46), 3)),
    "v_dense_sizes": list(itertools.combinations(range(15, 46), 3)),
    "r_dense_sizes": list(itertools.combinations(range(15, 46), 3)),

    # Fixed input dimension choice
    "do_fix_input_dim": [False],

    # Learning rate values
    "learning_rate": np.logspace(-4, -1, num=20)  # 20 values from 0.0001 to 0.1
}


import wandb
wandb.login(key="00a56e6e20885cb4a07fa9dac18639059ee73710")
# Initialize W&B
wandb.init(project="genetic_algorithm_optimization", entity="tourk-f")

# Initialize a population
def initialize_population(pop_size):
    print('initialize population')
    population = []
    for _ in range(pop_size):
        individual = {}
        for param, value in hyperparam_space.items():
            if param == "dense_sizes":
                # Handle 'dense_sizes' separately to pick only three random values for each category
                individual[param] = {k: random.choice(v) for k, v in value.items()}
            elif all(isinstance(elem, list) for elem in value):
                # Handle list of lists
                individual[param] = [random.choice(value[i]) for i in range(len(value))]
            else:
                # Handle simple list
                individual[param] = random.choice(value)
        population.append(individual)
        #print('individual', individual)
    return population



# Fitness function (model training and evaluation)
def evaluate_individual(individual, ind_num, generation):
    print('evaluate individual')
    # Clear any existing TensorFlow graph
    tf.keras.backend.clear_session()

    try:
        # Build the model with the individual's hyperparameters
        model = construct_model_2023(window_size=window_size,
                                       filter_sizes=list(individual['filter_sizes']),
                                       kernel_sizes=list(individual['kernel_sizes']),
                                       dilations=list(individual['dilations']),
                                       num_channels=individual['num_channels'],
                                       batch_norm_insertion_pts=list(individual['batch_norm_insertion_pts']),
                                       sp_dense_sizes=list(individual['sp_dense_sizes']),
                                       ss_dense_sizes=list(individual['ss_dense_sizes']),
                                       v_dense_sizes=list(individual['v_dense_sizes']),
                                       r_dense_sizes=list(individual['r_dense_sizes']),
                                       do_fix_input_dim=individual['do_fix_input_dim'])


        # Compile the model with the individual's learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=individual['learning_rate'])
        model.compile(loss=[data_loading_util.custom_loss, 'binary_crossentropy',
                            data_loading_util.custom_loss, data_loading_util.custom_loss_for_ramp],
                    loss_weights=[0.25, 0.25, 0.25, 0.25],
                    optimizer=optimizer)

        # Define callbacks
        filename = f'individual_{ind_num}_generation_{generation}.h5'
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
        mc = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

        tqdm_callback = TQDMProgressBar()
        history = model.fit(x=x_train, 
                        y=[y_sp_train, y_ss_train, y_v_train, y_r_train],
                        batch_size=32, 
                        epochs=30, 
                        validation_data=(x_valid, [y_sp_valid, y_ss_valid, y_v_valid, y_r_valid]),
                        callbacks=[es, mc, tqdm_callback],  # Add the custom callback
                        verbose=0)  # Set verbose to 0 to disable the default progress bar

        # Evaluate the model using the validation loss
        val_loss = model.evaluate(x_valid, [y_sp_i_valid, y_ss_i_valid, y_v_i_valid, y_r_i_valid, y_sp_c_valid, y_ss_c_valid], verbose=0)
        print(f"Individual {ind_num} in generation {generation} achieved validation loss: {val_loss}")
        # Log the individual's performance
        wandb.log({"generation": generation, 
                "individual_num": ind_num, 
                "val_loss": val_loss[0], 
                **individual})
        return val_loss[0]  # Assuming val_loss is a list and we're interested in the first value (overall loss)
    except ValueError as e:
        if str(e) == 'layers shrink the cnn too much':
            print(f"Configuration for individual {ind_num} in generation {generation} is not viable (CNN layers shrink too much). Assigning high validation loss.")
            return float('inf')  # Assign an infinite loss to indicate a poor configuration
        else:
            raise  # Re-raise the exception if it's not the specific one we're catching


# Selection
def select_parents(population, fitness_scores, num_parents):
    print('select parents')
    parents = np.array(population)[np.argsort(fitness_scores)[:num_parents]].tolist()  # Select the ones with lowest loss
    return parents

# Crossover
def crossover(parent1, parent2):
    #print('crossover')
    child = {}
    for param in hyperparam_space:
        if param == "dense_sizes":
            child[param] = {k: random.choice([parent1[param][k], parent2[param][k]]) for k in parent1[param]}
        else:
            child[param] = random.choice([parent1[param], parent2[param]])
    #print('child', child)
    return child

'''# Mutation
def mutate(individual):
    param_to_mutate = random.choice(list(hyperparam_space.keys()))
    individual[param_to_mutate] = random.choice(hyperparam_space[param_to_mutate])
    return individual'''

def mutate(individual):
    param_to_mutate = random.choice(list(hyperparam_space.keys()))

    try:
        # Handle 'dense_sizes' separately
        if param_to_mutate == "dense_sizes":
            for key in hyperparam_space["dense_sizes"]:
                individual[param_to_mutate][key] = random.choice(hyperparam_space[param_to_mutate][key])
        else:
            individual[param_to_mutate] = random.choice(hyperparam_space[param_to_mutate])

    except KeyError as e:
        # Log the error and relevant information
        print(f"KeyError occurred with key: {param_to_mutate}")
        print(f"Error details: {e}")
        print(f"Current individual: {individual}")
        # Handle the error appropriately

    return individual


# Genetic algorithm
def genetic_algorithm(pop_size, generations, num_parents):
    population = initialize_population(pop_size)

    for gen in range(generations):
        print(f"Starting generation {gen + 1}")

        # Evaluate individuals
        fitness_scores = [evaluate_individual(ind, ind_num, gen + 1) for ind_num, ind in enumerate(population)]

        # Logging best performance in the current generation
        min_loss = min(fitness_scores)
        best_individual = population[np.argmin(fitness_scores)]
        print(f"Best individual in generation {gen + 1}: {best_individual} with loss: {min_loss}")

        # Select parents
        parents = select_parents(population, fitness_scores, num_parents)

        # Generate next generation
        next_generation = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            next_generation.append(mutate(child))

        population = next_generation

        print(f"Generation {gen + 1} completed.\n")

    # Return best individual
    best_idx = np.argmin(fitness_scores)
    best_hyperparameters = population[best_idx]
    print(f"Best hyperparameters after {generations} generations: {best_hyperparameters}")
    # Log the best individual of the generation
    best_individual = population[np.argmin(fitness_scores)]
    min_loss = min(fitness_scores)
    wandb.log({"generation": gen + 1, 
                "best_val_loss": min_loss, 
                **best_individual})
    return best_hyperparameters

# Run the genetic algorithm
best_hyperparameters = genetic_algorithm(pop_size=10, generations=20, num_parents=4)
