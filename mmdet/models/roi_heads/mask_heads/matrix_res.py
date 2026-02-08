import torch
import torch.nn.functional as F


def matrix_reconstruction(Z1, S, mask):
    """
    Matrix reconstruction process.

    Args:
        Z1 (torch.Tensor): RGB feature tensor of shape (B, F, C1).
        S (torch.Tensor): Temporal self-similarity matrix of shape (B, H, F, F).
        mask (torch.Tensor): Mask of shape (B, F) with 0s and 1s.

    Returns:
        torch.Tensor: Reconstructed feature tensor ZS of shape (B, F).
    """
    # Step 1: Compute ZT by averaging Z1 along the channel dimension
    ZT = Z1.mean(dim=-1)  # Shape: (B, F)

    # Step 2: Compute SB by averaging S along the second dimension
    SB = S.mean(dim=1)  # Shape: (B, F, F)
    print(SB)

    # Set diagonal elements of SB to zero
    diag_indices = torch.arange(SB.size(-1))
    SB[:, diag_indices, diag_indices] = 0
    print(SB)

    # Normalize each row in SB so the sum equals 1
    SN = F.normalize(SB, p=1, dim=-1)  # Shape: (B, F, F)

    # Step 3: Reconstruct ZT using SN
    ZT_prime = torch.bmm(ZT.unsqueeze(1), SN).squeeze(1)  # Shape: (B, F)

    # Step 4: Calculate ZS using the mask
    ZS = mask * ZT + (1 - mask) * ZT_prime  # Shape: (B, F)

    return ZS


import numpy as np


def reconstruct_matrix(Z1, S):
    # Step 1: Compute the mean value of the RGB feature tensor Z1 across the channel dimension
    ZT = np.mean(Z1, axis=2)

    # Calculate the mean value of the temporal self-similarity matrix S along the second dimension
    SB = np.mean(S, axis=1)

    # Set the diagonal elements of each F x F matrix in SB to zero
    for i in range(SB.shape[0]):
        np.fill_diagonal(SB[i], 0)

    # Normalize the remaining elements in each matrix so that the sum of each row equals 1
    SN = SB / np.sum(SB, axis=-1, keepdims=True)

    # Step 2: Perform the reconstruction of the target feature ZT based on the normalized temporal self-similarity matrix SN
    ZT_prime = np.dot(ZT, SN)

    # Step 3: Randomly generate a mask M
    B, F = ZT.shape
    M = np.random.randint(0, 2, size=(B, F))

    # Step 4: Calculate the reconstruction source ZS
    ZS = M * ZT + (1 - M) * ZT_prime

    return ZS


# Example usage
B = 1
F = 4
C1 = 3
H = 1

# Generate example data
Z1 = np.random.rand(B, F, C1)
S = np.random.rand(B, H, F, F)

# Call the function with the example data
reconstructed_ZS = reconstruct_matrix(Z1, S)
print("Reconstructed ZS:", reconstructed_ZS)


# # Example usage
# B, F, C1, H = 1, 4, 3, 2  # Example dimensions
# Z1 = torch.randn(B, F, C1)
# S = torch.randn(B, H, F, F)
# mask = torch.randint(0, 2, (B, F)).float()  # Random binary mask
#
# ZS = matrix_reconstruction(Z1, S, mask)
# print("Reconstructed ZS:", ZS)
