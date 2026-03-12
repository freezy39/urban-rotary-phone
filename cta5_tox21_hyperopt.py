import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import deepchem as dc

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

tf.compat.v1.disable_eager_execution()

# ---------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

# ---------------------------------------------------
# Load Tox21 task 0 only
# ---------------------------------------------------
def load_tox21_task0():
    _, (train, valid, test), _ = dc.molnet.load_tox21()

    train_X, train_y, train_w = train.X, train.y[:, 0], train.w[:, 0]
    valid_X, valid_y, valid_w = valid.X, valid.y[:, 0], valid.w[:, 0]
    test_X, test_y, test_w = test.X, test.y[:, 0], test.w[:, 0]

    return train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w

# ---------------------------------------------------
# Baseline random forest
# ---------------------------------------------------
def run_random_forest_baseline():
    train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w = load_tox21_task0()

    sklearn_model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=50,
        random_state=456
    )

    print("About to fit RandomForest on train set.")
    sklearn_model.fit(train_X, train_y)

    train_y_pred = sklearn_model.predict(train_X)
    valid_y_pred = sklearn_model.predict(valid_X)
    test_y_pred = sklearn_model.predict(test_X)

    train_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
    valid_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    test_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)

    print(f"Weighted train Classification Accuracy: {train_score:.6f}")
    print(f"Weighted valid Classification Accuracy: {valid_score:.6f}")
    print(f"Weighted test Classification Accuracy: {test_score:.6f}")

    return {
        "train_acc": train_score,
        "valid_acc": valid_score,
        "test_acc": test_score
    }

# ---------------------------------------------------
# Fully connected network eval function
# ---------------------------------------------------
def eval_tox21_hyperparams(
    n_hidden=50,
    n_layers=1,
    learning_rate=0.001,
    dropout_keep_prob=0.5,
    n_epochs=20,
    batch_size=100,
    weight_positives=True,
    seed=456,
    run_tensorboard=False
):
    set_all_seeds(seed)
    tf.compat.v1.reset_default_graph()

    print("---------------------------------------------")
    print("Model hyperparameters")
    print(f"n_hidden = {n_hidden}")
    print(f"n_layers = {n_layers}")
    print(f"learning_rate = {learning_rate}")
    print(f"n_epochs = {n_epochs}")
    print(f"batch_size = {batch_size}")
    print(f"weight_positives = {weight_positives}")
    print(f"dropout_keep_prob = {dropout_keep_prob}")
    print(f"seed = {seed}")
    print("---------------------------------------------")

    train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w = load_tox21_task0()

    d = train_X.shape[1]
    graph = tf.Graph()

    with graph.as_default():
        with tf.compat.v1.name_scope("placeholders"):
            x = tf.compat.v1.placeholder(tf.float32, (None, d), name="x")
            y = tf.compat.v1.placeholder(tf.float32, (None,), name="y")
            w = tf.compat.v1.placeholder(tf.float32, (None,), name="w")
            keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")

        x_hidden = x
        input_dim = d

        for layer in range(n_layers):
            with tf.compat.v1.name_scope(f"layer_{layer}"):
                W = tf.Variable(
                    tf.random.normal((input_dim, n_hidden), stddev=0.1),
                    name=f"W_{layer}"
                )
                b = tf.Variable(
                    tf.zeros((n_hidden,)),
                    name=f"b_{layer}"
                )
                x_hidden = tf.nn.relu(tf.matmul(x_hidden, W) + b)
                x_hidden = tf.nn.dropout(x_hidden, rate=1 - keep_prob)
                input_dim = n_hidden

        with tf.compat.v1.name_scope("output"):
            W_out = tf.Variable(
                tf.random.normal((input_dim, 1), stddev=0.1),
                name="W_out"
            )
            b_out = tf.Variable(
                tf.zeros((1,)),
                name="b_out"
            )
            y_logit = tf.matmul(x_hidden, W_out) + b_out
            y_one_prob = tf.sigmoid(y_logit)
            y_pred = tf.round(y_one_prob)

        with tf.compat.v1.name_scope("loss"):
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_logit,
                labels=y_expand
            )

            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy

            l = tf.reduce_mean(entropy)

        with tf.compat.v1.name_scope("optim"):
            train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(l)

        with tf.compat.v1.name_scope("summaries"):
            tf.compat.v1.summary.scalar("loss", l)
            merged = tf.compat.v1.summary.merge_all()

        hyperparam_str = (
            f"d-{d}-hidden-{n_hidden}-layers-{n_layers}-lr-{learning_rate}"
            f"-epochs-{n_epochs}-batch-{batch_size}-weightpos-{weight_positives}"
            f"-dropout-{dropout_keep_prob}-seed-{seed}"
        )

        logdir = os.path.join("tensorboard_logs", hyperparam_str)
        os.makedirs("tensorboard_logs", exist_ok=True)

        loss_history = []
        N = train_X.shape[0]

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            train_writer = None
            if run_tensorboard:
                train_writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)

            step = 0
            for epoch in range(n_epochs):
                pos = 0
                while pos < N:
                    batch_X = train_X[pos:pos + batch_size]
                    batch_y = train_y[pos:pos + batch_size]
                    batch_w = train_w[pos:pos + batch_size]

                    feed_dict = {
                        x: batch_X,
                        y: batch_y,
                        w: batch_w,
                        keep_prob: dropout_keep_prob
                    }

                    _, loss_val, summary = sess.run([train_op, l, merged], feed_dict=feed_dict)
                    loss_history.append(loss_val)

                    if step % 20 == 0:
                        print(f"epoch {epoch}, step {step}, loss: {loss_val:.6f}")

                    if train_writer is not None:
                        train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

            valid_y_pred = sess.run(
                y_pred,
                feed_dict={x: valid_X, keep_prob: 1.0}
            ).reshape(-1)

            test_y_pred = sess.run(
                y_pred,
                feed_dict={x: test_X, keep_prob: 1.0}
            ).reshape(-1)

        valid_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
        test_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)

    print(f"Valid Weighted Classification Accuracy: {valid_score:.6f}")
    print(f"Test Weighted Classification Accuracy: {test_score:.6f}")

    return {
        "valid_acc": valid_score,
        "test_acc": test_score,
        "loss_history": loss_history,
        "logdir": logdir
    }

# ---------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------
def hyperparameter_search():
    # choose a manageable grid
    hidden_list = [50, 100]
    layer_list = [1, 2]
    lr_list = [0.001, 0.0005]
    dropout_list = [0.5, 0.8]
    epoch_list = [20, 30]
    batch_list = [100]
    weight_pos_list = [True, False]

    repeats = 3
    base_seed = 456

    results = []
    best_result = None

    for n_hidden in hidden_list:
        for n_layers in layer_list:
            for learning_rate in lr_list:
                for dropout_keep_prob in dropout_list:
                    for n_epochs in epoch_list:
                        for batch_size in batch_list:
                            for weight_positives in weight_pos_list:
                                valid_scores = []
                                test_scores = []

                                for rep in range(repeats):
                                    seed = base_seed + rep

                                    out = eval_tox21_hyperparams(
                                        n_hidden=n_hidden,
                                        n_layers=n_layers,
                                        learning_rate=learning_rate,
                                        dropout_keep_prob=dropout_keep_prob,
                                        n_epochs=n_epochs,
                                        batch_size=batch_size,
                                        weight_positives=weight_positives,
                                        seed=seed,
                                        run_tensorboard=False
                                    )

                                    valid_scores.append(out["valid_acc"])
                                    test_scores.append(out["test_acc"])

                                avg_valid = float(np.mean(valid_scores))
                                std_valid = float(np.std(valid_scores))
                                avg_test = float(np.mean(test_scores))

                                result = {
                                    "n_hidden": n_hidden,
                                    "n_layers": n_layers,
                                    "learning_rate": learning_rate,
                                    "dropout_keep_prob": dropout_keep_prob,
                                    "n_epochs": n_epochs,
                                    "batch_size": batch_size,
                                    "weight_positives": weight_positives,
                                    "avg_valid_acc": avg_valid,
                                    "std_valid_acc": std_valid,
                                    "avg_test_acc": avg_test
                                }

                                results.append(result)

                                print("\n=== COMBINATION SUMMARY ===")
                                print(json.dumps(result, indent=2))

                                if best_result is None or avg_valid > best_result["avg_valid_acc"]:
                                    best_result = result

    print("\n==============================")
    print("BEST HYPERPARAMETER SETTING")
    print(json.dumps(best_result, indent=2))
    print("==============================")

    with open("tox21_hyperparam_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_result, results

# ---------------------------------------------------
# Plot best loss curve from one final run
# ---------------------------------------------------
def run_best_and_save_curve(best_result):
    out = eval_tox21_hyperparams(
        n_hidden=best_result["n_hidden"],
        n_layers=best_result["n_layers"],
        learning_rate=best_result["learning_rate"],
        dropout_keep_prob=best_result["dropout_keep_prob"],
        n_epochs=best_result["n_epochs"],
        batch_size=best_result["batch_size"],
        weight_positives=best_result["weight_positives"],
        seed=456,
        run_tensorboard=True
    )

    plt.figure(figsize=(8, 5))
    plt.plot(out["loss_history"])
    plt.title("Best Model Training Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("best_model_loss_curve.png", dpi=200)
    plt.show()

    print(f"TensorBoard logdir for best run: {out['logdir']}")

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    print("Running Random Forest baseline...")
    baseline = run_random_forest_baseline()

    print("\nStarting hyperparameter search...")
    best_result, all_results = hyperparameter_search()

    print("\nRunning best model again to save TensorBoard logs and loss curve...")
    run_best_and_save_curve(best_result)

    print("\nDone.")