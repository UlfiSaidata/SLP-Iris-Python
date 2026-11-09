import pandas as pd
import math
import matplotlib.pyplot as plt

# ==========================================================
# 1. MEMBACA DATA EXCEL
# ==========================================================

file_excel = "PMM-Tugas SLP - Ulfi.xlsx"

data = pd.read_excel(
    file_excel,
    sheet_name="Data",
    header=None
)

# Memberikan nama kolom
data.columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
    "Species"
]

# Menghapus baris yang seluruhnya kosong
data = data.dropna(how="all").reset_index(drop=True)

print("=" * 60)
print("DATA IRIS")
print("=" * 60)

print(data.head())

print("\nJumlah data:", len(data))


# ==========================================================
# 2. MENENTUKAN TARGET
# ==========================================================

data["Target"] = data["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1
})

# Memastikan tidak ada kelas yang gagal dipetakan
if data["Target"].isna().any():
    raise ValueError(
        "Ada label Species yang tidak dikenali. "
        "Periksa nama kelas pada Excel."
    )

data["Target"] = data["Target"].astype(int)


# ==========================================================
# 3. MEMBAGI DATA TRAINING DAN VALIDATION
# ==========================================================

setosa = data[data["Species"] == "Iris-setosa"]

versicolor = data[data["Species"] == "Iris-versicolor"]

# 40 Setosa + 40 Versicolor untuk training
train = pd.concat([
    setosa.iloc[:40],
    versicolor.iloc[:40]
]).reset_index(drop=True)

# 10 Setosa + 10 Versicolor untuk validation
validation = pd.concat([
    setosa.iloc[40:50],
    versicolor.iloc[40:50]
]).reset_index(drop=True)

print("\nJumlah Training:", len(train))
print("Jumlah Validation:", len(validation))


# ==========================================================
# 4. PARAMETER AWAL
# ==========================================================

bias = 0.5

w1 = 0.5
w2 = 0.5
w3 = 0.5
w4 = 0.5

learning_rate = 0.1

epochs = 5


# ==========================================================
# 5. FUNGSI SIGMOID
# ==========================================================

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# ==========================================================
# 6. MENYIMPAN METRIK
# ==========================================================

train_mse_history = []
train_accuracy_history = []

val_mse_history = []
val_accuracy_history = []


# ==========================================================
# 7. TRAINING SLP
# ==========================================================

for epoch in range(1, epochs + 1):

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch}")
    print("=" * 70)

    total_error = 0
    correct = 0

    # ------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------

    for i in range(len(train)):

        x1 = train.loc[i, "SepalLengthCm"]
        x2 = train.loc[i, "SepalWidthCm"]
        x3 = train.loc[i, "PetalLengthCm"]
        x4 = train.loc[i, "PetalWidthCm"]

        target = train.loc[i, "Target"]

        # Forward propagation
        z = bias + w1*x1 + w2*x2 + w3*x3 + w4*x4

        output = sigmoid(z)

        # Prediksi
        prediksi = 1 if output > 0.5 else 0

        # Error
        error = output - target

        # SSE
        sse = error ** 2

        # Delta bias
        delta_bias = (
            2 * (output - target)
            * (1 - output)
            * output
        )

        # Delta bobot
        delta_w1 = delta_bias * x1
        delta_w2 = delta_bias * x2
        delta_w3 = delta_bias * x3
        delta_w4 = delta_bias * x4

        # Update bias
        bias_baru = bias - learning_rate * delta_bias

        # Update bobot
        w1_baru = w1 - learning_rate * delta_w1
        w2_baru = w2 - learning_rate * delta_w2
        w3_baru = w3 - learning_rate * delta_w3
        w4_baru = w4 - learning_rate * delta_w4

        # Metrik dihitung dari output sebelum update
        total_error += sse

        if prediksi == target:
            correct += 1

        # Menampilkan detail sampel
        print(
            f"Sample {i+1:02d} | "
            f"Target={target} | "
            f"Output={output:.6f} | "
            f"Prediksi={prediksi} | "
            f"Error={error:.6f} | "
            f"SSE={sse:.6f}"
        )

        # Update parameter untuk sampel berikutnya
        bias = bias_baru

        w1 = w1_baru
        w2 = w2_baru
        w3 = w3_baru
        w4 = w4_baru

    # ------------------------------------------------------
    # METRIK TRAINING
    # ------------------------------------------------------

    train_mse = total_error / len(train)

    train_accuracy = correct / len(train) * 100

    train_mse_history.append(train_mse)

    train_accuracy_history.append(train_accuracy)

    print("\nTraining MSE:", train_mse)

    print("Training Accuracy:", train_accuracy, "%")

    print("\nBobot setelah training:")

    print("Bias:", bias)
    print("W1:", w1)
    print("W2:", w2)
    print("W3:", w3)
    print("W4:", w4)


    # ======================================================
    # 8. VALIDATION
    # ======================================================

    print("\n--- VALIDATION ---")

    total_error_val = 0

    correct_val = 0

    for i in range(len(validation)):

        x1 = validation.loc[i, "SepalLengthCm"]
        x2 = validation.loc[i, "SepalWidthCm"]
        x3 = validation.loc[i, "PetalLengthCm"]
        x4 = validation.loc[i, "PetalWidthCm"]

        target = validation.loc[i, "Target"]

        # Menggunakan bobot hasil training
        z = bias + w1*x1 + w2*x2 + w3*x3 + w4*x4

        output = sigmoid(z)

        prediksi = 1 if output > 0.5 else 0

        error = output - target

        sse = error ** 2

        total_error_val += sse

        if prediksi == target:
            correct_val += 1

        print(
            f"Validation {i+1:02d} | "
            f"Target={target} | "
            f"Output={output:.6f} | "
            f"Prediksi={prediksi} | "
            f"SSE={sse:.6f}"
        )

    # Metrik validation
    val_mse = total_error_val / len(validation)

    val_accuracy = correct_val / len(validation) * 100

    val_mse_history.append(val_mse)

    val_accuracy_history.append(val_accuracy)

    print("\nValidation MSE:", val_mse)

    print("Validation Accuracy:", val_accuracy, "%")


# ==========================================================
# 9. RINGKASAN HASIL SELURUH EPOCH
# ==========================================================

print("\n" + "=" * 70)
print("RINGKASAN HASIL")
print("=" * 70)

hasil = pd.DataFrame({
    "Epoch": range(1, epochs + 1),
    "Training MSE": train_mse_history,
    "Training Accuracy (%)": train_accuracy_history,
    "Validation MSE": val_mse_history,
    "Validation Accuracy (%)": val_accuracy_history
})

print(hasil.to_string(index=False))


# ==========================================================
# 10. GRAFIK MSE
# ==========================================================

epoch_numbers = range(1, epochs + 1)

plt.figure(figsize=(10, 5))

plt.plot(
    epoch_numbers,
    train_mse_history,
    marker="o",
    label="Training MSE"
)

plt.plot(
    epoch_numbers,
    val_mse_history,
    marker="s",
    label="Validation MSE"
)

plt.title("Training MSE vs Validation MSE")

plt.xlabel("Epoch")

plt.ylabel("Mean Squared Error")

plt.xticks(epoch_numbers)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ==========================================================
# 11. GRAFIK AKURASI
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    epoch_numbers,
    train_accuracy_history,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epoch_numbers,
    val_accuracy_history,
    marker="s",
    label="Validation Accuracy"
)

plt.title("Training Accuracy vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.xticks(epoch_numbers)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()