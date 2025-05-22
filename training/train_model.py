def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history
