import torch
from base.dgd.latent import RepresentationLayer
from base.plotting.plot_cv2 import plot_latent_space, plot_gene, plot_mirna
from base.plotting.plot import plot_gene_recons, plot_mirna_recons

from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error, r2_score, pearson_corrcoef, spearman_corrcoef, mean_squared_log_error

from tqdm import tqdm
import wandb

# Training functions
def train_midgd(dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), nepochs=100, fold=None, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=1310, wandb_log=False, early_stopping=50, is_plot=True):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim

        tlen_mirna = len(train_loader.dataset)*dgd.decoder.n_out_features_mirna
        vlen_mirna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mirna
        tlen_mrna = len(train_loader.dataset)*dgd.decoder.n_out_features_mrna
        vlen_mrna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mrna
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)

    # Train Representation Layer Initialization
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    # Test/Validation Representation Layer Initialization
    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    # Optimizer Initialization
    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    # Metrics logger initialization
    loss_tab = {"epoch":[],
                "train_recon_mirna":[],"train_recon_mrna":[], 
                "test_recon_mirna":[], "test_recon_mrna":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))
    
    # Early stopping
    best_loss = 10000
    best_epoch = -1

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            
            loss_tab["train_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["train_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss =  mrna_recon_loss + mirna_recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            # Get data
            scaling = torch.mean(train_loader.dataset.mirna_data, axis=1)
            mirna_recon, _ = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = train_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()

        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon_mrna"].append(0.)
        loss_tab["test_recon_mirna"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_msle"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, _, lib_mrna, _, index) in validation_loader:
            mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="midgd"
            )
            loss_tab["test_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["test_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = mrna_recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            scaling = torch.mean(validation_loader.dataset.mirna_data, axis=1)
            mirna_recon, _ = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = validation_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon_mrna"][-1] /= vlen_mrna
        loss_tab["test_recon_mirna"][-1] /= vlen_mirna
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon_mirna: {loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mrna: {loss_tab['test_recon_mrna'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",3000), train_loader.dataset.label, color_mapping, epoch, dataset="Train")
                plot_latent_space(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.label, color_mapping, epoch, dataset="Validation")
                plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train")
                plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation")
                plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train")
                plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation")
            
        if wandb_log: 
            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_recon_mrna": loss_tab["train_recon_mrna"][-1],
                "train_recon_mirna": loss_tab["train_recon_mirna"][-1],
                "train_gmm": loss_tab["train_gmm"][-1],
                "train_mse": loss_tab["train_mse"][-1],
                "train_mae": loss_tab["train_mae"][-1],
                "train_r2": loss_tab["train_r2"][-1],
                "train_spearman": loss_tab["train_spearman"][-1],
                "train_pearson": loss_tab["train_pearson"][-1],
                "train_msle": loss_tab["train_msle"][-1],
                "test_recon_mrna": loss_tab["test_recon_mrna"][-1],
                "test_gmm": loss_tab["test_gmm"][-1],
                "test_mse": loss_tab["test_mse"][-1],
                "test_mae": loss_tab["test_mae"][-1],
                "test_r2": loss_tab["test_r2"][-1],
                "test_spearman": loss_tab["test_spearman"][-1],
                "test_pearson": loss_tab["test_pearson"][-1],
                "test_msle": loss_tab["test_msle"][-1]})

        # Early stopping
        if early_stopping:
            if loss_tab['test_recon_mirna'][-1] < best_loss:
                best_loss = loss_tab['test_recon_mirna'][-1]
                best_epoch = epoch
                checkpoint = dgd
            elif epoch - best_epoch > early_stopping:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
    
    if early_stopping:
        dgd = checkpoint        
    
    # Training done!
    return loss_tab


def train_midgd_all(dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), nepochs=100, fold=None, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=1310, wandb_log=False, early_stopping=50, is_plot=True):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim

        tlen_mirna = len(train_loader.dataset)*dgd.decoder.n_out_features_mirna
        vlen_mirna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mirna
        tlen_mrna = len(train_loader.dataset)*dgd.decoder.n_out_features_mrna
        vlen_mrna = len(validation_loader.dataset)*dgd.decoder.n_out_features_mrna
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)

    # Train Representation Layer Initialization
    Ntrain=len(train_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    # Test/Validation Representation Layer Initialization
    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    # Optimizer Initialization
    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    # Metrics logger initialization
    loss_tab = {"epoch":[],
                "train_recon_mirna":[],"train_recon_mrna":[], 
                "test_recon_mirna":[], "test_recon_mrna":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))
    
    # Early stopping
    best_loss = 100000
    best_epoch = -1

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon_mirna"].append(0.)
        loss_tab["train_recon_mrna"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train() # Training mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            
            loss_tab["train_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["train_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss =  mrna_recon_loss + mirna_recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            # Get data
            scaling = torch.mean(train_loader.dataset.mirna_data, axis=1)
            mirna_recon, _ = dgd.forward(dgd.train_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = train_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_pearson"][-1] += mirna_pearson.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["train_msle"][-1] += mirna_msle.item()

        loss_tab["train_recon_mirna"][-1] /= tlen_mirna
        loss_tab["train_recon_mrna"][-1] /= tlen_mrna
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon_mrna"].append(0.)
        loss_tab["test_recon_mirna"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_msle"].append(0.)

        # Train the validation representation layer only using mRNA data
        val_rep_optimizer.zero_grad()
        dgd.eval() # Validation mode
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in validation_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss_tab["test_recon_mirna"][-1] += mirna_recon_loss.item()
            loss_tab["test_recon_mrna"][-1] += mrna_recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            
            loss = mirna_recon_loss + mrna_recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            scaling = torch.mean(validation_loader.dataset.mirna_data, axis=1)
            mirna_recon, _ = dgd.forward(dgd.val_rep())
            mirna_recon = mirna_recon * scaling.unsqueeze(1).to(device)
            mirna_data = validation_loader.dataset.mirna_data.to(device)
            # Get subset
            mirna_recon_tpm = mirna_recon[:,subset]
            mirna_data_tpm = mirna_data[:,subset]
            # Calculate metrics
            mirna_mse = mean_squared_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mse"][-1] += mirna_mse.item()
            mirna_mae = mean_absolute_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_mae"][-1] += mirna_mae.item()
            mirna_r2 = r2_score(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_r2"][-1] += mirna_r2.item()
            mirna_spearman = spearman_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_spearman"][-1] += mirna_spearman.item()
            mirna_pearson = pearson_corrcoef(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_pearson"][-1] += mirna_pearson.item()
            mirna_msle = mean_squared_log_error(mirna_recon_tpm, mirna_data_tpm)
            loss_tab["test_msle"][-1] += mirna_msle.item()

        loss_tab["test_recon_mrna"][-1] /= vlen_mrna
        loss_tab["test_recon_mirna"][-1] /= vlen_mirna
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon_mirna: {loss_tab['train_recon_mirna'][-1]}", 
                  f"train_recon_mrna: {loss_tab['train_recon_mrna'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon_mirna: {loss_tab['test_recon_mirna'][-1]}",
                  f"test_recon_mrna: {loss_tab['test_recon_mrna'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if is_plot:
            if plot>=0 and (epoch)%plot==0:
                plot_latent_space(*dgd.get_latent_space_values("train",3000), train_loader.dataset.label, color_mapping, epoch, dataset="Train")
                plot_latent_space(*dgd.get_latent_space_values("val",3000), validation_loader.dataset.label, color_mapping, epoch, dataset="Validation")
                plot_mirna(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train")
                plot_mirna(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation")
                plot_gene(dgd, train_loader, sample_index, epoch, device, fold=fold, type="Train")
                plot_gene(dgd, validation_loader, sample_index, epoch, device, fold=fold, type="Validation")
            
        if wandb_log: 
            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_recon_mrna": loss_tab["train_recon_mrna"][-1],
                "train_recon_mirna": loss_tab["train_recon_mirna"][-1],
                "train_gmm": loss_tab["train_gmm"][-1],
                "train_mse": loss_tab["train_mse"][-1],
                "train_mae": loss_tab["train_mae"][-1],
                "train_r2": loss_tab["train_r2"][-1],
                "train_spearman": loss_tab["train_spearman"][-1],
                "train_pearson": loss_tab["train_pearson"][-1],
                "train_msle": loss_tab["train_msle"][-1],
                "test_recon_mrna": loss_tab["test_recon_mrna"][-1],
                "test_gmm": loss_tab["test_gmm"][-1],
                "test_mse": loss_tab["test_mse"][-1],
                "test_mae": loss_tab["test_mae"][-1],
                "test_r2": loss_tab["test_r2"][-1],
                "test_spearman": loss_tab["test_spearman"][-1],
                "test_pearson": loss_tab["test_pearson"][-1],
                "test_msle": loss_tab["test_msle"][-1]})
        
        # Early stopping
        if early_stopping:
            if loss_tab['test_recon_mirna'][-1] < best_loss:
                best_loss = loss_tab['test_recon_mirna'][-1]
                best_epoch = epoch
                checkpoint = dgd
            elif epoch - best_epoch > early_stopping:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
    
    if early_stopping:
        dgd = checkpoint   
        
    # Training done!
    return loss_tab


def train_dgd_mrna(dgd, train_loader, validation_loader, device,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), nepochs=100, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=1310, wandb_log=False):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train()
        for (mrna_data, lib_mrna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=mrna_data.to(device),  # Pass both mRNA and miRNA data
                scale=lib_mrna.unsqueeze(1).to(device),  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="mrna"
            )

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            scaling = torch.mean(train_loader.dataset.data, axis=1)
            recon = dgd.forward(dgd.train_rep()) * scaling.unsqueeze(1).to(device)
            data = train_loader.dataset.data.to(device)
            # Get subset
            recon_tpm = recon[:,subset]
            data_tpm = data[:,subset]
            # Calculate metrics
            mse = mean_squared_error(recon_tpm, data_tpm)
            loss_tab["train_mse"][-1] += mse.item()
            mae = mean_absolute_error(recon_tpm, data_tpm)
            loss_tab["train_mae"][-1] += mae.item()
            r2 = r2_score(recon_tpm, data_tpm)
            loss_tab["train_r2"][-1] += r2.item()
            spearman = spearman_corrcoef(recon_tpm, data_tpm)
            loss_tab["train_spearman"][-1] += spearman.item()
            pearson = pearson_corrcoef(recon_tpm, data_tpm)
            loss_tab["train_pearson"][-1] += pearson.item()
            msle = mean_squared_log_error(recon_tpm, data_tpm)
            loss_tab["train_msle"][-1] += msle.item()

        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_msle"].append(0.)

        val_rep_optimizer.zero_grad()
        dgd.eval()
        for (mrna_data, lib_mrna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=mrna_data.to(device),  # Pass both mRNA and miRNA data
                scale=lib_mrna.unsqueeze(1).to(device),  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="mrna"
            )

            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            scaling = torch.mean(validation_loader.dataset.data, axis=1)
            recon = dgd.forward(dgd.val_rep()) * scaling.unsqueeze(1).to(device)
            data = validation_loader.dataset.data.to(device)
            # Get subset
            recon_tpm = recon[:,subset]
            data_tpm = data[:,subset]
            # Calculate metrics
            mse = mean_squared_error(recon_tpm, data_tpm)
            loss_tab["test_mse"][-1] += mse.item()
            mae = mean_absolute_error(recon_tpm, data_tpm)
            loss_tab["test_mae"][-1] += mae.item()
            r2 = r2_score(recon_tpm, data_tpm)
            loss_tab["test_r2"][-1] += r2.item()
            spearman = spearman_corrcoef(recon_tpm, data_tpm)
            loss_tab["test_spearman"][-1] += spearman.item()
            pearson = pearson_corrcoef(recon_tpm, data_tpm)
            loss_tab["test_pearson"][-1] += pearson.item()
            msle = mean_squared_log_error(recon_tpm, data_tpm)
            loss_tab["test_msle"][-1] += msle.item()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon: {loss_tab['train_recon'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon: {loss_tab['test_recon'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train", 3000), train_loader.dataset.label, color_mapping, epoch, dataset="Train")
            plot_latent_space(*dgd.get_latent_space_values("val", 3000), validation_loader.dataset.label, color_mapping, epoch, dataset="Validation")
            plot_gene_recons(dgd, train_loader, sample_index, epoch, device, type="Train")
            plot_gene_recons(dgd, validation_loader, sample_index, epoch, device, type="Validation")

    return loss_tab


def train_dgd_mirna(dgd, train_loader, validation_loader, device, train_dataset, val_dataset,
                learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
                weight_decay=0., betas=(0.9, 0.999), nepochs=100, pr=1, plot=10, reduction_type="sum", 
                sample_index=[0,11,22,33], subset=1310, wandb_log=False):
    # Normalization factor
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        vlen=len(validation_loader)
        tlen_gmm=len(train_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'], weight_decay=weight_decay, betas=betas)
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'], weight_decay=weight_decay, betas=betas)
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'], weight_decay=weight_decay, betas=betas)

    loss_tab = {"epoch":[],
                "train_recon":[],"test_recon":[],
                "train_gmm":[],"test_gmm":[], 
                "train_mse":[], "test_mse":[],
                "train_mae":[], "test_mae":[],
                "train_r2":[], "test_r2":[],
                "train_spearman":[], "test_spearman":[],
                "test_pearson":[], "train_pearson":[],
                "train_msle":[], "test_msle":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["epoch"].append(epoch)
        loss_tab["train_recon"].append(0.)
        loss_tab["train_gmm"].append(0.)
        loss_tab["train_mse"].append(0.)
        loss_tab["train_mae"].append(0.)
        loss_tab["train_r2"].append(0.)
        loss_tab["train_spearman"].append(0.)
        loss_tab["train_pearson"].append(0.)
        loss_tab["train_msle"].append(0.)

        train_rep_optimizer.zero_grad()
        dgd.train()
        for (mirna_data, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: 
                gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=mirna_data.to(device),  # Pass both mRNA and miRNA data
                scale=lib_mirna.unsqueeze(1).to(device),  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="mirna"
            )  

            loss_tab["train_recon"][-1] += recon_loss.item()
            loss_tab["train_gmm"][-1] += gmm_loss.item()

            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: 
                gmm_optimizer.step()
        train_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            scaling = torch.mean(train_loader.dataset.data, axis=1)
            recon = dgd.forward(dgd.train_rep()) * scaling.unsqueeze(1).to(device)
            data = train_loader.dataset.data.to(device)
            # Get subset
            recon_tpm = recon[:,subset]
            data_tpm = data[:,subset]
            # Calculate metrics
            mse = mean_squared_error(recon_tpm, data_tpm)
            loss_tab["train_mse"][-1] += mse.item()
            mae = mean_absolute_error(recon_tpm, data_tpm)
            loss_tab["train_mae"][-1] += mae.item()
            r2 = r2_score(recon_tpm, data_tpm)
            loss_tab["train_r2"][-1] += r2.item()
            spearman = spearman_corrcoef(recon_tpm, data_tpm)
            loss_tab["train_spearman"][-1] += spearman.item()
            pearson = pearson_corrcoef(recon_tpm, data_tpm)
            loss_tab["train_pearson"][-1] += pearson.item()
            msle = mean_squared_log_error(recon_tpm, data_tpm)
            loss_tab["train_msle"][-1] += msle.item()

        loss_tab["train_recon"][-1] /= tlen
        loss_tab["train_gmm"][-1] /= tlen_gmm

        # Validation step
        loss_tab["test_recon"].append(0.)
        loss_tab["test_gmm"].append(0.)
        loss_tab["test_mse"].append(0.)
        loss_tab["test_mae"].append(0.)
        loss_tab["test_r2"].append(0.)
        loss_tab["test_spearman"].append(0.)
        loss_tab["test_pearson"].append(0.)
        loss_tab["test_msle"].append(0.)

        val_rep_optimizer.zero_grad()
        dgd.eval()
        for (mirna_data, lib_mirna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=mirna_data.to(device),  # Pass both mRNA and miRNA data
                scale=lib_mirna.unsqueeze(1).to(device),  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="mirna"
            )

            loss_tab["test_recon"][-1] += recon_loss.item()
            loss_tab["test_gmm"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        # Calculate metrics
        with torch.inference_mode():
            scaling = torch.mean(validation_loader.dataset.data, axis=1)
            recon = dgd.forward(dgd.val_rep()) * scaling.unsqueeze(1).to(device)
            data = validation_loader.dataset.data.to(device)
            # Get subset
            recon_tpm = recon[:,subset]
            data_tpm = data[:,subset]
            # Calculate metrics
            mse = mean_squared_error(recon_tpm, data_tpm)
            loss_tab["test_mse"][-1] += mse.item()
            mae = mean_absolute_error(recon_tpm, data_tpm)
            loss_tab["test_mae"][-1] += mae.item()
            r2 = r2_score(recon_tpm, data_tpm)
            loss_tab["test_r2"][-1] += r2.item()
            spearman = spearman_corrcoef(recon_tpm, data_tpm)
            loss_tab["test_spearman"][-1] += spearman.item()
            pearson = pearson_corrcoef(recon_tpm, data_tpm)
            loss_tab["test_pearson"][-1] += pearson.item()
            msle = mean_squared_log_error(recon_tpm, data_tpm)
            loss_tab["test_msle"][-1] += msle.item()

        loss_tab["test_recon"][-1] /= vlen
        loss_tab["test_gmm"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch,
                  f"train_recon: {loss_tab['train_recon'][-1]}", 
                  f"train_gmm: {loss_tab['train_gmm'][-1]}",
                  f"train_mse: {loss_tab['train_mse'][-1]}",
                  f"train_mae: {loss_tab['train_mae'][-1]}",
                  f"train_r2: {loss_tab['train_r2'][-1]}",
                  f"train_spearman: {loss_tab['train_spearman'][-1]}",
                  f"train_pearson: {loss_tab['train_pearson'][-1]}",
                  f"train_msle: {loss_tab['train_msle'][-1]}")
            print(epoch, 
                  f"test_recon: {loss_tab['test_recon'][-1]}", 
                  f"test_gmm: {loss_tab['test_gmm'][-1]}",
                  f"test_mse: {loss_tab['test_mse'][-1]}",
                  f"test_mae: {loss_tab['test_mae'][-1]}",
                  f"test_r2: {loss_tab['test_r2'][-1]}",
                  f"test_spearman: {loss_tab['test_spearman'][-1]}",
                  f"test_pearson: {loss_tab['test_pearson'][-1]}",
                  f"test_msle: {loss_tab['test_msle'][-1]}")
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train", 3000), train_loader.dataset.label, color_mapping, epoch, dataset="Train")
            plot_latent_space(*dgd.get_latent_space_values("val", 3000), validation_loader.dataset.label, color_mapping, epoch, dataset="Validation")
            plot_mirna_recons(dgd, train_loader, sample_index, epoch, device, type="Train")
            plot_mirna_recons(dgd, validation_loader, sample_index, epoch, device, type="Validation")

    return loss_tab