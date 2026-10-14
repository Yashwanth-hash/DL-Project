
# --------------------------------------------------
# 6. BUILD THE ORIGINAL LSTM CLASSIFIER
# --------------------------------------------------

print("\nBuilding original LSTM model...")

lstm_model = models.Sequential([
    layers.Input(
        shape=(SEQUENCE_LENGTH, features.shape[-1])
    ),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.4),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(CLASSES), activation="softmax"),
])

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

lstm_model.summary()


# --------------------------------------------------
# 7. TRAIN
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "drowsiness_lstm_best.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)


# --------------------------------------------------
# 8. EVALUATE
# --------------------------------------------------

print("\nEvaluating on test participants...")

test_probabilities = lstm_model.predict(
    X_test, batch_size=32
)
y_pred = np.argmax(test_probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0,
)
print("\nClassification report:")
print(report)

cm = confusion_matrix(
    y_test, y_pred, labels=[0, 1, 2]
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN-LSTM Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "cnn_lstm_confusion_matrix.png",
    dpi=150,
)
plt.close()

lstm_model.save(OUTPUT_DIR / "drowsiness_cnn_lstm.keras")

with open(RESULTS_DIR / "cnn_lstm_report.txt", "w") as f:
    f.write(f"Test accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)

print("\nTraining complete.")
print("Model:", OUTPUT_DIR / "drowsiness_cnn_lstm.keras")
print("Results:", RESULTS_DIR / "cnn_lstm_report.txt")
print("Confusion matrix:", RESULTS_DIR / "cnn_lstm_confusion_matrix.png")
