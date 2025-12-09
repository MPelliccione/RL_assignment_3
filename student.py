import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.normal import Normal

# Hyperparameters
LATENT_SIZE = 32
HIDDEN_SIZE = 256
ACTION_SIZE = 3
GAUSSIANS = 5

class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(2*2*256, LATENT_SIZE)
        self.fc_logvar = nn.Linear(2*2*256, LATENT_SIZE)
        
        self.decoder_input = nn.Linear(LATENT_SIZE, 2*2*256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 2*2*256)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        d = self.decoder_input(z)
        d = d.view(-1, 256, 2, 2)
        recon = self.decoder(d)
        return recon, mu, logvar

class MDRNN(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super(MDRNN, self).__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        
        self.lstm = nn.LSTM(latents + actions, hiddens)
        self.fc = nn.Linear(hiddens, gaussians * (latents * 2 + 1))

    def forward(self, actions, latents, hidden):
        ins = torch.cat([actions, latents], dim=-1)
        outs, hidden = self.lstm(ins, hidden)
        return self.fc(outs), hidden

    def get_mixture_params(self, y):
        seq_len, batch_size, _ = y.size()
        stride = self.gaussians * self.latents
        
        mus = y[:, :, :stride]
        mus = mus.view(seq_len, batch_size, self.gaussians, self.latents)
        
        sigmas = y[:, :, stride:2*stride]
        sigmas = sigmas.view(seq_len, batch_size, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)
        
        logpi = y[:, :, 2*stride:]
        logpi = logpi.view(seq_len, batch_size, self.gaussians)
        logpi = F.log_softmax(logpi, dim=-1)
        
        return mus, sigmas, logpi

class Controller(nn.Module):
    def __init__(self, latents, hiddens, actions):
        super(Controller, self).__init__()
        self.fc = nn.Linear(latents + hiddens, actions)

    def forward(self, z, h):
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

class Policy(nn.Module):
    continuous = True 

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.vae = VAE(device).to(device)
        self.mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, HIDDEN_SIZE, GAUSSIANS).to(device)
        self.controller = Controller(LATENT_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(device)
        self.hidden = None

    def forward(self, x):
        return x
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device) / 255.0
        state = state.permute(2, 0, 1).unsqueeze(0)
        state = F.interpolate(state, size=(64, 64), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            _, mu, _ = self.vae(state)
            z = mu
            
            if self.hidden is None:
                h = torch.zeros(1, 1, HIDDEN_SIZE).to(self.device)
                c = torch.zeros(1, 1, HIDDEN_SIZE).to(self.device)
                self.hidden = (h, c)
            
            h_t = self.hidden[0].squeeze(0)
            action = self.controller(z, h_t)
            action = action.cpu().numpy()[0]
            
            z_in = z.unsqueeze(0)
            a_in = torch.tensor(action).float().to(self.device).view(1, 1, -1)
            _, self.hidden = self.mdrnn.lstm(torch.cat([a_in, z_in], dim=-1), self.hidden)
            
        return action

    def train(self):
        try:
            import cma
        except ImportError:
            print("CMA library not found. Please install it to train.")
            return

        print("Starting training pipeline...")
        
        # 1. Collect Rollouts
        print("Collecting rollouts...")
        env = gym.make('CarRacing-v2', continuous=True)
        rollouts = []
        n_rollouts = 100 # Increase for better results
        
        for i in range(n_rollouts):
            obs, _ = env.reset()
            done = False
            episode_obs = []
            episode_act = []
            while not done:
                action = env.action_space.sample()
                # Resize obs for storage to save space/time
                obs_small = F.interpolate(torch.from_numpy(obs).float().permute(2,0,1).unsqueeze(0), size=(64,64)).squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
                episode_obs.append(obs_small)
                episode_act.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            rollouts.append((np.array(episode_obs), np.array(episode_act)))
            if (i+1) % 10 == 0:
                print(f"Collected {i+1} rollouts")
        
        # 2. Train VAE
        print("Training VAE...")
        optimizer = torch.optim.Adam(self.vae.parameters())
        # Flatten rollouts
        all_obs = np.concatenate([r[0] for r in rollouts], axis=0)
        dataset = TensorDataset(torch.from_numpy(all_obs).permute(0, 3, 1, 2).float() / 255.0)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(5): # Increase epochs
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.vae(x)
                
                # Loss
                recon_loss = F.mse_loss(recon, x, reduction='sum')
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"VAE Epoch {epoch+1}, Loss: {total_loss/len(dataset)}")
            
        # 3. Train MDRNN
        print("Training MDRNN...")
        # Generate z sequences
        z_seqs = []
        action_seqs = []
        with torch.no_grad():
            for obs, act in rollouts:
                obs_torch = torch.from_numpy(obs).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                _, mu, _ = self.vae(obs_torch)
                z_seqs.append(mu)
                action_seqs.append(torch.from_numpy(act).float().to(self.device))
        
        # Prepare data for MDRNN
        # Input: z_t, a_t. Target: z_{t+1}
        # We train on sequences.
        mdrnn_opt = torch.optim.Adam(self.mdrnn.parameters())
        
        for epoch in range(5): # Increase epochs
            total_loss = 0
            for z_seq, a_seq in zip(z_seqs, action_seqs):
                # z_seq: (T, 32), a_seq: (T, 3)
                # Inputs: z[:-1], a[:-1]
                # Targets: z[1:]
                if len(z_seq) < 2: continue
                
                ins_z = z_seq[:-1].unsqueeze(1) # (T-1, 1, 32)
                ins_a = a_seq[:-1].unsqueeze(1) # (T-1, 1, 3)
                targets = z_seq[1:].unsqueeze(1) # (T-1, 1, 32)
                
                mdrnn_opt.zero_grad()
                hidden = (torch.zeros(1, 1, HIDDEN_SIZE).to(self.device), torch.zeros(1, 1, HIDDEN_SIZE).to(self.device))
                
                outs, _ = self.mdrnn(ins_a, ins_z, hidden)
                mus, sigmas, logpi = self.mdrnn.get_mixture_params(outs)
                
                # Loss: NLL
                # targets: (T-1, 1, 32)
                # mus: (T-1, 1, 5, 32)
                targets = targets.unsqueeze(2).expand(-1, -1, GAUSSIANS, -1)
                
                dist = Normal(mus, sigmas)
                log_probs = dist.log_prob(targets) # (T-1, 1, 5, 32)
                log_probs = torch.sum(log_probs, dim=3) # Sum over latent dim -> (T-1, 1, 5)
                
                # Combine with mixture weights
                log_probs = log_probs + logpi
                loss = -torch.logsumexp(log_probs, dim=2).mean()
                
                loss.backward()
                mdrnn_opt.step()
                total_loss += loss.item()
            print(f"MDRNN Epoch {epoch+1}, Loss: {total_loss/len(z_seqs)}")

        # 4. Train Controller with CMA-ES
        print("Training Controller with CMA-ES...")
        # Flatten controller parameters
        params = self.controller.state_dict()
        # We only optimize weights, not biases for simplicity or both.
        # Let's optimize all.
        # Helper to get/set params
        def get_flat_params(model):
            return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()
        
        def set_flat_params(model, flat_params):
            idx = 0
            for p in model.parameters():
                n = p.data.numel()
                p.data.copy_(torch.from_numpy(flat_params[idx:idx+n]).view_as(p.data).to(self.device))
                idx += n
                
        init_params = get_flat_params(self.controller)
        es = cma.CMAEvolutionStrategy(init_params, 0.1, {'popsize': 16})
        
        for gen in range(5): # Increase generations
            solutions = es.ask()
            rewards = []
            for sol in solutions:
                set_flat_params(self.controller, sol)
                # Evaluate
                total_reward = 0
                obs, _ = env.reset()
                self.hidden = None # Reset hidden
                done = False
                while not done:
                    action = self.act(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                rewards.append(total_reward)
            
            es.tell(solutions, [-r for r in rewards]) # CMA minimizes
            print(f"Generation {gen+1}, Best Reward: {max(rewards)}")
            
        # Set best params
        best_params = es.result[0]
        set_flat_params(self.controller, best_params)
        self.save()
        print("Training complete.")

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        if os.path.exists('model.pt'):
            self.load_state_dict(torch.load('model.pt', map_location=self.device))
        self.vae.eval()
        self.mdrnn.eval()
        self.controller.eval()

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
