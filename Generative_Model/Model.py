import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck_size = 128  # this might be worth to play around with ...
        self.encoder = Encoder(bottleneck_size=self.bottleneck_size)
        self.decoder = Decoder(bottleneck_size=self.bottleneck_size)
        self.prior_distribution = torch.distributions.Normal(0, 1) # this is the prior over the latent space P(z)

    def forward(self,
                target_data=None,  # during training this should be a batch of target data.
                # During inference simply leave this out to sample unconditionally.
                noise_scale_during_inference=0.8,  # might be worth to play around with this ...
                device="cpu"
                ):
        if target_data is not None:
            # run the encoder
            means, log_variance = self.encoder(target_data) #

            # convert means and log_variance to sample
            z = means + log_variance * self.prior_distribution.sample(means.shape)

        else:
            z = torch.randn(self.bottleneck_size).to(device).unsqueeze(0) * noise_scale_during_inference    

        # run the decoder
        reconstructions_of_targets = self.decoder(z)

        if target_data is not None:
            # calculate the losses
            predicted_distribution = torch.distributions.Normal(means, log_variance.exp())
            kl_loss = torch.distributions.kl_divergence(predicted_distribution, self.prior_distribution).mean()
            reconstruction_loss = torch.nn.functional.mse_loss(reconstructions_of_targets, target_data, reduction="mean")
            return reconstructions_of_targets, kl_loss, reconstruction_loss

        return reconstructions_of_targets



class Encoder(torch.nn.Module):
    def __init__(self, bottleneck_size):
        """
        The input to the encoder will have the shape (batch_size, 1, 28, 28)
        The output should be a batch of vectors of the bottleneck_size
        """
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 200)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(200, bottleneck_size)

    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x.view(-1, 28*28))))
        h = self.dropout(F.relu(self.fc2(h)))
        mean = self.fc(h)
        log_variance = self.fc(h)
        return  mean, log_variance



class Decoder(nn.Module):
    """
    The input of the decoder will be a batch of fixed size vectors with the bottleneck_size as dimensionality
    The output of the decoder should have the shape (batch_size, 1, 28, 28)
    """
    def __init__(self, bottleneck_size):
        super().__init__()
        self.lstm = nn.LSTM(bottleneck_size, 512, num_layers=2, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(512,800)
        self.fc2 = nn.Linear(800, 28*28)
        self.relu = nn.ReLU()

    def forward(self, z):
        # LSTM expects input of shape (batch_size, seq_len, input_size)
        # Here we treat the bottleneck vector as a sequence of length 1
        z = z.unsqueeze(1)
        lstm_out, _ = self.lstm(z)
        # We only care about the final output of the LSTM
        lstm_out = lstm_out[:, -1, :]
        h = self.relu(self.fc1(lstm_out))
        out = torch.sigmoid(self.fc2(h))
        return out.view(-1, 1, 28, 28)

