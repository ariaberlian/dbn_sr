import multiprocessing
import logging
from deep_belief_network.dbn.tensorflow.models import SupervisedDBNRegression
from utils.data_processing import DataProcessing
from utils.image_file_util import *
from utils.scoring import *
from utils.visualizer import *
import pandas as pd
from tqdm import tqdm
import gc
import multiprocessing
import time
import os

log_folder = "experiment_logs"
os.makedirs(log_folder, exist_ok=True)  # Create folder if it doesn't exist

def configure_logging(idx):
    idx += 1
    """Configure logging to use a unique file per iteration in a specific folder."""
    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Define the log file path for this specific iteration
    log_file = os.path.join(log_folder, f"experiment_{idx}.log")
    
    # Set up logging with a unique filename for this iteration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=False),
            logging.StreamHandler()
        ]
    )

excel_name = "Experimentation.xlsx"
exp_df = pd.read_excel(excel_name, engine="openpyxl")


def test(i, row, model_name, model, interpolation_factor, patch_size, stride, data_train_size, dp: DataProcessing):
    test_var = row[f"test{i}"]
    test_ref_var = row[f"test{i}_ref"]
    
    test_image = load_image(f"test/{test_var}")
    test_reference_image = load_image(f"test/{test_ref_var}")
    
    model = model.load(f"model/{model_name}.h5")
    
    logging.info(f"Testing model: {model_name}...")

    interpolated_test = dp.interpolate(test_image, interpolation_factor)

    test_patches = dp.get_patches(interpolated_test, patch_size, stride)
    norm, int_test_dct_ex = dp.normalize_for_rbm(test_patches[:test_patches.shape[0]])
    if data_train_size < test_patches.shape[0]:
        norm = dp.normalize_for_rbm(test_patches[:data_train_size])

    test_patches_flat = dp.preprocess_for_rbm(norm)
    norm = None  # Explicitly delete the norm variable after use

    result_flat = model.predict(test_patches_flat)
    result_flat = dp.proccess_output(test_patches_flat, result_flat, interpolation_factor)

    result_patches, recons_dct_ex = dp.inverse_preprocess(
        result_flat, (patch_size[0], patch_size[1], 3)
    )

    reconstruct_image = dp.reconstruct_from_patches(
        result_patches, original_shape=test_reference_image.shape, patch_size=patch_size, stride=stride
    )

    norm, refs_dct_ex = dp.normalize_for_rbm(dp.get_patches(test_reference_image, patch_size, stride))
    refs_flat = dp.preprocess_for_rbm(norm)

    psnr_baseline = calculate_psnr(test_reference_image, interpolated_test)
    ssim_baseline = calculate_ssim(test_reference_image, interpolated_test) * 100
    psnr = calculate_psnr(test_reference_image, reconstruct_image)
    ssim = calculate_ssim(test_reference_image, reconstruct_image) * 100
    rmse = calculate_rmse(refs_flat, result_flat)

    logging.info(f"PSNR {i} Baseline value: {psnr_baseline:,.3f} dB")
    logging.info(f"SSIM {i} Baseline value: {ssim_baseline:,.3f} %")
    logging.info(f"PSNR {i}: {psnr:,.3f} dB")
    logging.info(f"SSIM {i}: {ssim:,.3f} %")
    logging.info(f"RMSE {i}: {rmse}")

    # Cleanup: explicitly delete all variables
    del test_image, test_reference_image, interpolated_test, test_patches, test_patches_flat
    del result_flat, result_patches, reconstruct_image, refs_flat, norm
    gc.collect()

    return psnr_baseline, ssim_baseline, psnr, ssim, rmse


def experiments(idx, row):
    configure_logging(idx)
    try:
        # Read parameters from Excel for this row
        train = row["train"]
        fine_image_path = row["fine_tuning"]
        fine_label = row["label"]
        train_resolution = row['train_res']
        interpolation_factor = row['factor']
        patch_size = (row['patch_size'], row['patch_size'])
        stride = (row['stride'], row['stride'])
        data_train_size = row["data_train_size"]

        logging.info(f"Fine Tuning Image: {fine_image_path}")

        lr = 0.001
        epoch = 500
        epoch_fine = 100
        layers = [int(layer) for layer in str(row['layers']).split(",")]
        batch_size = row['batch_size']
        activation_function = row["activation_function"]

        model_name = f"model_{train}_ft{fine_image_path}_{train_resolution}_x{interpolation_factor}_p{patch_size[0]}_s{stride[0]}_l{layers}"
        logging.info(f"Training model: {model_name}")

        # #### Model Configuration ####
        dbn = SupervisedDBNRegression(
                    hidden_layers_structure=layers,
                    batch_size=batch_size,
                    learning_rate_rbm=lr,
                    n_epochs_rbm=epoch,
                    activation_function=activation_function,
                    optimization_algorithm='sgd',
                    learning_rate=lr,
                    n_iter_backprop=epoch_fine,
        )

        #### Load Data and Preprocess ####
        dp = DataProcessing()

        training_image = load_image(f"train/{train}")
        train_patches = dp.get_patches(training_image, patch_size=patch_size, stride=stride)
        norm, _ = dp.normalize_for_rbm(train_patches[:train_patches.shape[0]])
        if data_train_size < train_patches.shape[0]:
            norm = dp.normalize_for_rbm(train_patches[:data_train_size])

        X_pretrain = dp.preprocess_for_rbm(norm)

        del train_patches, training_image, norm
        gc.collect()

        fine_image = dp.interpolate(load_image(f'train/{fine_image_path}'), 2)
        label = load_image(f'train/{fine_label}')

        fine_patches = dp.get_patches(fine_image, patch_size=patch_size, stride=stride)
        norm, _ = dp.normalize_for_rbm(fine_patches[:fine_patches.shape[0]])
        if data_train_size < fine_patches.shape[0]:
            norm = dp.normalize_for_rbm(fine_patches[:data_train_size])
        X = dp.preprocess_for_rbm(norm)

        del fine_patches, fine_image, norm
        gc.collect()

        label_patches = dp.get_patches(label, patch_size=patch_size, stride=stride)
        norm, _ = dp.normalize_for_rbm(label_patches[:label_patches.shape[0]])
        if data_train_size < label_patches.shape[0]:
            norm = dp.normalize_for_rbm(label_patches[:data_train_size])
        y = dp.preprocess_for_rbm(norm)

        del label, label_patches, norm
        gc.collect()

        #### Train the Model ####
        if not(os.path.exists(f"model/{model_name}.h5")):
            logging.info(f"Starting training for model: {model_name}")
            start_time = time.time()
            dbn.fit(X_pretrain, X, y)
            end_time = time.time()

            logging.info(f"Training time: {end_time - start_time:.2f} seconds")
            dbn.save(f"model/{model_name}.h5")

            logging.info(f"Model has been saved: {model_name}")
            exp_df.at[idx, "training_time"] = (end_time - start_time)

            del X_pretrain, X, y
            gc.collect()
        else:
            logging.info("Model already exists!")

        #### Testing and Saving Results ####
        for i in range(1, 6):
            psnr_baseline, ssim_baseline, psnr, ssim, rmse = test(i, row, model_name, dbn, interpolation_factor, patch_size, stride, data_train_size, dp)
            # Save results to DataFrame
            exp_df.at[idx, f"b_psnr{i}"] = psnr_baseline
            exp_df.at[idx, f"b_ssim{i}"] = ssim_baseline
            exp_df.at[idx, f"psnr{i}"] = psnr
            exp_df.at[idx, f"ssim{i}"] = ssim
            exp_df.at[idx, f"rmse{i}"] = rmse
            exp_df.to_excel(excel_name, index=False)

        logging.info(f"Finished processing row {idx + 1}/{len(exp_df)} with model {model_name}")

    except Exception as e:
        logging.error(f"Error processing row {idx + 1}: {e}")

    finally:
        # Ensure all variables are deleted in the end
        del train, fine_label, train_resolution, interpolation_factor
        del patch_size, stride, data_train_size, lr, epoch, layers, batch_size, activation_function, model_name
        gc.collect()

def run_experiments(idx, row):
    data = (idx, row)

    p = multiprocessing.Process(target=experiments, args=data)
    p.daemon = True
    p.start()
    p.join() 


if __name__ == "__main__":
    for idx, row in tqdm(exp_df.iterrows(), total=len(exp_df), desc="Processing Rows"):
        row_dict = row.to_dict()  # Convert row to dictionary for serialization
        run_experiments(idx, row_dict)