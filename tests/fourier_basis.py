import matplotlib.pyplot as plt
from srffwfs.fourier_basis import compute_fourier_basis

n_px = 20
n_mode = -1

fourier_basis, labels = compute_fourier_basis(n_px, remove_piston=False)

fourier_basis_flat = fourier_basis.reshape(fourier_basis.shape[0], -1)

print(f"fourier basis shape: {fourier_basis.shape}")
for label in labels:
    print(label)

# visualize a given Fourier mode
plt.figure()
plt.imshow(fourier_basis[n_mode])
plt.title(f"Fourier mode: {labels[n_mode]}")

# check orthonormality
plt.figure()
plt.imshow(fourier_basis_flat @ fourier_basis_flat.T)
plt.title("Orthonormality check")

plt.show()
