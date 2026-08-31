from pc_rnn_control.robot.policy import *


def main(
    data: str | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save: str = "lstm_inverse_dynamics.pth",
    seed: int = 42,
    seq_len: int = 1,
    num_past_actions_as_input: int = 1,
    num_past_positions_as_input: int = 1,
    use_sequence: bool = False,
    plot_dataset: bool = False,
    use_elman: bool = False,
    index_obs_frames=None,
    learn_with_pose_error=False,
    pos_min=None,
    pos_max=None,
    act_min=None,
    act_max=None
):
    """Train the inverse dynamics LSTM.

    Parameters are passed directly instead of argparse. Provide `data` as the
    path to a recording .npz.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if data:
        full_ds = InverseDynamicsWindowDataset(
            data,
            seq_len=seq_len,
            num_past_actions_as_input=num_past_actions_as_input,
            num_past_positions_as_input=num_past_positions_as_input,
            use_sequence=use_sequence,
            plot_dataset=plot_dataset,
            index_obs_frames=index_obs_frames,
            learn_with_pose_error=learn_with_pose_error,
            pos_min=pos_min,
            pos_max=pos_max,
            action_min=act_min,
            action_max=act_max
        )
    else:
        raise RuntimeError("A data path must be provided")

    # Create train/val split on dataset indices (80/20)
    M = len(full_ds)
    split_idx = int(0.8 * M)

    train_idx = list(range(0, split_idx))
    val_idx = list(range(split_idx, M))

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    # Determine input and output dims from the dataset
    # train_ds is a Subset of InverseDynamicsWindowDataset; access underlying arrays via full_ds
    input_feat_dim = full_ds.inputs.shape[-1]
    action_dim = full_ds.targets.shape[-1]
    input_dim = input_feat_dim

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    device_t = torch.device(device)
    model = LSTMInverseDynamics(
        input_dim=input_dim,
        action_dim=action_dim,
        use_elman_rnn=use_elman,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):

        t0 = time.time()

        # Training loop
        model.train()

        total_loss = 0.0
        loss_func = nn.MSELoss()

        for batch in tqdm(train_loader, desc="train", leave=False):
            input_batch = batch[0].to(device_t)
            targets_batch = batch[1].to(device_t)

            optimizer.zero_grad()

            preds = model(input_batch)  # (Batch, Seq_len, action_dim)
            preds_last = preds[:, -1, :]
            loss = loss_func(preds_last, targets_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        val_loss = eval_epoch(model, val_loader, device_t)
        val_losses.append(val_loss)
        t1 = time.time()

        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={t1-t0:.1f}s"
        )

        # Save the model with the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "params": {
                        "data": data,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "lr": lr,
                        "device": device,
                        "seed": seed,
                        "seq_len": seq_len,
                        "min_pos_training": full_ds.min_pos,
                        "max_pos_training": full_ds.max_pos,
                        "min_act_training": full_ds.min_action,
                        "max_act_training": full_ds.max_action,
                        "use_elman": use_elman,
                    },
                },
                save,
            )
            print(f"Saved best model to {save}")

    print("Training finished")

    # --- plotting and prediction on full dataset (train + val) ---
    if plt is None:
        print("matplotlib not available; skipping plotting of results")
        return

    # Build a DataLoader for the full dataset (preserve ordering)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)

    # If a saved checkpoint exists, try loading it for prediction/inference
    if os.path.exists(save):
        try:
            params = {
                "input_dim": input_dim,
                "action_dim": action_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "device": device_t,
            }
            loaded_model = load_checkpoint(save, params)
            model = loaded_model
            print(f"Loaded checkpoint from {save} for prediction")
        except Exception as e:
            print(
                f"Warning: failed to load checkpoint {save}: {e}; using in-memory model"
            )

    model.eval()
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="predict", leave=False):
            states = batch[0].to(device_t)
            targets = batch[1].to(device_t)
            preds = model(states)
            preds_last = preds[:, -1, :]
            all_preds.append(preds_last.cpu().numpy())
            all_tgts.append(targets.cpu().numpy())

    if len(all_preds) == 0:
        print("No predictions available; skipping plotting")
        return

    preds_np = np.concatenate(all_preds, axis=0)
    tgts_np = np.concatenate(all_tgts, axis=0)

    # Create plots: (1) loss curves, (2) predicted vs true
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Loss curves
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axes[0].plot(range(1, len(val_losses) + 1), val_losses, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()
    axes[0].set_title("Training and validation loss")

    # Predicted vs true: show a time-series overlay for the first few action dims
    num_samples = preds_np.shape[0]
    action_dim = preds_np.shape[1]

    # Time-series overlay: plot the first up to 3 action dimensions (true = solid, pred = dashed)
    max_plot_dims = min(3, action_dim)
    for a in range(max_plot_dims):
        axes[1].plot(tgts_np[:, a], label=f"true_{a}", alpha=0.8)
        axes[1].plot(preds_np[:, a], linestyle="--", label=f"pred_{a}", alpha=0.8)
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("action value")
    axes[1].set_title(
        f"Predicted (dashed) vs True (solid) actions — first {max_plot_dims} dims"
    )
    axes[1].legend()

    plt.tight_layout()

    plt.show()

    return model, full_ds


def test_model_on_data(
    model_checkpoint_path,
    test_data_path,
    seq_len=1,
    num_past_actions_as_input=1,
    num_past_positions_as_input=1,
    use_sequence=False,
    index_obs_frames=None,
    learn_with_pose_error=False,
    device="cpu",
    plot_dataset=False,  # Added this parameter
):
    """
    Independent test function to perform inference on a specific test dataset
    using a saved checkpoint.
    """
    print(
        f"\n--- Starting Inference on Test Set: {os.path.basename(test_data_path)} ---"
    )

    # 1. Load Checkpoint and Extract Params
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Checkpoint not found at {model_checkpoint_path}")
        return

    device_t = torch.device(device)
    ckpt = torch.load(model_checkpoint_path, map_location=device_t, weights_only=False)

    # Extract training parameters to reconstruct exact architecture
    ckpt_params = ckpt.get("params", {})
    hidden_size = ckpt_params.get("hidden_size", 128)
    num_layers = ckpt_params.get("num_layers", 2)
    dropout = ckpt_params.get("dropout", 0.0)
    use_elman = ckpt_params.get("use_elman", False)

    # 2. Reconstruct Dataset to get dynamic dimensions
    test_dataset = InverseDynamicsWindowDataset(
        test_data_path,
        seq_len=seq_len,
        num_past_actions_as_input=num_past_actions_as_input,
        num_past_positions_as_input=num_past_positions_as_input,
        use_sequence=use_sequence,
        plot_dataset=plot_dataset,
        index_obs_frames=index_obs_frames,
        learn_with_pose_error=learn_with_pose_error,
    )

    # Important: Determine dims from the freshly built dataset
    input_dim = test_dataset.inputs.shape[-1]
    action_dim = test_dataset.targets.shape[-1]

    # 3. Initialize Model
    model = LSTMInverseDynamics(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_elman_rnn=use_elman,
    ).to(device_t)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 4. Perform Inference
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Running Inference"):
            # x shape: [Batch, Seq_len, Input_dim]
            x = x.to(device_t)
            preds = model(x)

            # Use the prediction from the last time step in the sequence
            preds_last = preds[:, -1, :].cpu().numpy().flatten()
            all_preds.append(preds_last)
            all_targets.append(y.numpy().flatten())

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    # 5. Calculate Accuracy Metric
    test_mse = np.mean((preds_np - targets_np) ** 2)
    print(f"Test Inference MSE: {test_mse:.8f}")

    # 6. Final Visualization of Test Results
    plt.figure(figsize=(12, 6))
    colors = ["#e74c3c", "#085328", "#dbbd12"]
    max_dims = min(3, targets_np.shape[1])

    for i in range(max_dims):
        plt.plot(
            targets_np[:, i], color=colors[i], alpha=0.3, label=f"Motor {i+1} Target"
        )
        plt.plot(
            preds_np[:, i], color=colors[i], ls="--", label=f"Motor {i+1} Prediction"
        )

    plt.title(f"Test Set Inference: {os.path.basename(test_data_path)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Action Value")
    plt.legend(loc="upper right", ncol=max_dims)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_script_path)
    save_path = os.path.join(current_folder_path, "..", "data", "closed_loop")
    # Load actuations data dynamically_saturating_actuations
    freq = 10
    data_file = os.path.join(
        save_path, f"spiraling_babbling_{freq}_Hz_{length}_{version}_mvel0.5.npz"
    )

    # call main with hardcoded data path
    # save trained models under the repository's learned_models directory
    save_file = os.path.join(save_path, f"lstm_inverse_dynamics.pth")

    index_obs_frames = [1, 3, 5, 7]

    act_min = np.array([ 300., 900.,  100.])
    act_max = np.array([5000 , 6000, 6000])

    pos_max = None
    pos_min = None
    pos_min = None


    num_past_positions_as_input = 3
    num_past_actions_as_input = 3
    seq_len = 3
    use_sequence = True
    plot_dataset = True
    device = torch.device("cpu")
    use_elman = False
    learn_with_pose_error = False
    _, dataset = main(
       data=data_file,
       epochs=100,
       batch_size=64,
       seq_len=seq_len,
       use_elman=use_elman,
       hidden_size=128,
       dropout=0.25,
       num_layers=1,
       save=save_file,
       num_past_actions_as_input=num_past_actions_as_input,
       num_past_positions_as_input=num_past_positions_as_input,
       use_sequence=use_sequence,
       plot_dataset=plot_dataset,
       device=device,
       index_obs_frames=index_obs_frames,
       learn_with_pose_error=learn_with_pose_error,
       pos_max=pos_max,
       pos_min=pos_min,
       act_max=act_max,
       act_min=act_min
    )

    # Load another test dataset for online evaluation
    data_file = os.path.join(
        save_path, f"spiraling_babbling_{freq}_Hz_{length}_{version}_mvel0.5.npz"
    )

    # Test batch
    test_model_on_data(
        model_checkpoint_path=save_file,
        test_data_path=data_file,
        seq_len=seq_len,
        num_past_actions_as_input=num_past_actions_as_input,
        num_past_positions_as_input=num_past_positions_as_input,
        use_sequence=use_sequence,
        index_obs_frames=index_obs_frames,
        learn_with_pose_error=learn_with_pose_error,
        device="cpu", # or device
        plot_dataset=False
    )

