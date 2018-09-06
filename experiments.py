import os

#Specify one of {TimeSeriesData, BouncingBallsData, MovingMNISTData}
from data_loader import TimeSeriesData as DataLoader
from results_plotter import TimeSeriesPlotter as Plotter

from run import RunVAEInfiniteData as Run
from generative_models import RobustARGeneratorMeanFieldPosterior as GenerativeModel


predict_forward = False

# Training Parameters
pre_trained = False
# reduced as was getting an error before -- Ben
num_iterations = 1000
num_forward_iterations = 0

n_hidden_py = 50

seq_length = 100
n_samples = 50
batch_size = 50
training_size = 900
experiment_path = 'experiments/MFPost_InfData'


if __name__ == '__main__':
    os.makedirs('./{}/training_curves/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/output/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/data/'.format(experiment_path), exist_ok=True)
    os.makedirs('./{}/ckpt/'.format(experiment_path), exist_ok=True)

    if Run.__name__ == 'RunVAEInfiniteData':
        DataLoaderClass = DataLoader(seq_length = seq_length, training_size = batch_size*1000)
        data = None
        #data for plotting only:
        testing_data_clean, testing_data = DataLoaderClass.create_dataset()
        print('Data will be created inline during training.')

    elif DataLoader.__name__ == 'TimeSeriesData':
        print("Creating dataset...")
        DataLoaderClass = DataLoader(seq_length = seq_length, training_size = 2*training_size)   #also creates a test set of size training_size
        y, y_tilde = DataLoaderClass.create_dataset()
        data = y_tilde[:training_size,:]
        testing_data = y_tilde[training_size:, :]
        #data for plotting only:
        testing_data_clean = y[training_size:, :]
        print("Dataset created.")

    elif DataLoader.__name__ == 'BouncingBallsData':
        print("Creating dataset...")
        DataLoaderClass = DataLoader()
        data, testing_data = DataLoaderClass.create_dataset()
        print("Dataset created.")

    elif DataLoader.__name__ == 'MovingMNISTData':
        print("Creating dataset...")
        DataLoaderClass = DataLoader(training_size)
        data, testing_data = DataLoaderClass.create_dataset()
        print("Dataset created.")


    run = Run(data, testing_data, num_iterations, num_forward_iterations, seq_length, n_samples, batch_size,
              training_size, pre_trained, GenerativeModel, n_hidden_py, predict_forward, experiment_path, DataLoader)
    run.train()

    plotter = Plotter(testing_data, testing_data_clean, batch_size, seq_length, n_samples,
                    GenerativeModel, n_hidden_py, experiment_path)
    plotter.plot()
