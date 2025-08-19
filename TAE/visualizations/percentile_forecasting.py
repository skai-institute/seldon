import numpy as np
from TAE.datasets.lightcurve_dataset import enhanced_collate_fn

def percentile_forecast(experiment, batch):
        # Make a plot of reconstruction residual over batch
        indices = batch['raw_idx']
        percentiles = (np.arange(9)+1)/10
        resid = {'in':[], 'out':[], 'all':[], 'output':{}, 'batch':{}}
        for p in percentiles:
            samples = []
            for idx in indices:
                lc_idx, mode, lc_full, band_idx_full, length, class_name = experiment.data.dataset.get_raw_lightcurve(idx)
                if length < 10:
                    continue
                aug_length = int(length * p)
                aug_idx = np.arange(aug_length)
                lc_part = lc_full[aug_idx]
                band_idx_part = band_idx_full[aug_idx]
                sample = experiment.data.dataset.preprocess(lc_idx, lc_full, band_idx_full, length, lc_part, band_idx_part, aug_idx, mode, class_name)
                samples.append(sample)
            samples = enhanced_collate_fn(samples)
            out = experiment.model(
                x=samples["x_input"].to(experiment.device),
                t=samples["time_full"].to(experiment.device),
                band_idx_part=samples["band_idx"].to(experiment.device),
                band_idx_full=samples["band_idx_full"].to(experiment.device),
                mask=samples["in_sample_mask"].to(experiment.device),
                time_sorted=None,
                sequence_batch_mask=None,
            )
            
            residuals = (samples['flux_full'].to(experiment.device) - out["reconstructed"])/samples["flux_err_full"].to(experiment.device)
            residuals_in_sample = residuals[samples['in_sample_mask'].to(experiment.device)].view(-1)
            residuals_out_sample = residuals[~samples['in_sample_mask'].to(experiment.device) & samples["pad_mask_full"].to(experiment.device)].view(-1)
            residuals = residuals[samples["pad_mask_full"].to(experiment.device)].view(-1)
            resid['in'].append(residuals_in_sample.detach().cpu().numpy())
            resid['out'].append(residuals_out_sample.detach().cpu().numpy())
            resid['all'].append(residuals.detach().cpu().numpy())
            resid['output'][str(p)] = out
            resid['batch'][str(p)] = samples

        return resid
