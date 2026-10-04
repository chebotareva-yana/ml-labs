import numpy as np


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.root = None
        self.rng = None


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.rng = np.random.default_rng(self.random_state)

        self.root = self._grow_tree(
            X,
            y,
            depth=0
        )

        return self


    def _majority_class(self, y):
        return np.bincount(
            y,
            minlength=2
        ).argmax()


    def _get_feature_subset(self, n_features):
        if self.max_features == "sqrt":
            size = max(
                1,
                int(np.sqrt(n_features))
            )

        elif self.max_features == "log2":
            size = max(
                1,
                int(np.log2(n_features))
            )

        elif isinstance(self.max_features, int):
            size = min(
                self.max_features,
                n_features
            )

        else:
            size = n_features

        return self.rng.choice(
            n_features,
            size=size,
            replace=False
        )


    def _best_split(self, X, y, feature_indices):
        best_feature = None
        best_threshold = None
        best_gini = np.inf

        n_samples = len(y)

        for feature in feature_indices:
            values = X[:, feature]

            order = np.argsort(values)

            x_sorted = values[order]
            y_sorted = y[order]

            different = np.where(
                x_sorted[:-1] != x_sorted[1:]
            )[0]

            if len(different) == 0:
                continue

            cumulative_positive = np.cumsum(y_sorted)
            total_positive = cumulative_positive[-1]

            for i in different:
                n_left = i + 1
                n_right = n_samples - n_left

                if (
                    n_left < self.min_samples_leaf
                    or
                    n_right < self.min_samples_leaf
                ):
                    continue

                positive_left = cumulative_positive[i]
                negative_left = (
                    n_left - positive_left
                )

                positive_right = (
                    total_positive - positive_left
                )

                negative_right = (
                    n_right - positive_right
                )


                p_left = positive_left / n_left
                n_left_share = negative_left / n_left

                gini_left = (
                    1
                    - p_left ** 2
                    - n_left_share ** 2
                )


                p_right = positive_right / n_right
                n_right_share = negative_right / n_right

                gini_right = (
                    1
                    - p_right ** 2
                    - n_right_share ** 2
                )


                weighted_gini = (
                    n_left * gini_left
                    + n_right * gini_right
                ) / n_samples


                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature

                    best_threshold = (
                        x_sorted[i]
                        + x_sorted[i + 1]
                    ) / 2


        return (
            best_feature,
            best_threshold
        )


    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))


        stop = (
            n_classes == 1
            or n_samples < self.min_samples_split
            or (
                self.max_depth is not None
                and depth >= self.max_depth
            )
        )


        if stop:
            return Node(
                value=self._majority_class(y)
            )


        feature_indices = (
            self._get_feature_subset(
                n_features
            )
        )


        feature, threshold = self._best_split(
            X,
            y,
            feature_indices
        )


        if feature is None:
            return Node(
                value=self._majority_class(y)
            )


        left_mask = (
            X[:, feature] <= threshold
        )

        right_mask = ~left_mask


        if (
            left_mask.sum() == 0
            or right_mask.sum() == 0
        ):
            return Node(
                value=self._majority_class(y)
            )


        left = self._grow_tree(
            X[left_mask],
            y[left_mask],
            depth + 1
        )

        right = self._grow_tree(
            X[right_mask],
            y[right_mask],
            depth + 1
        )


        return Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right
        )


    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(
                x,
                node.left
            )

        return self._predict_one(
            x,
            node.right
        )


    def predict(self, X):
        X = np.asarray(X, dtype=float)

        return np.array([
            self._predict_one(
                x,
                self.root
            )
            for x in X
        ])


class RandomForest:
    def __init__(
        self,
        n_estimators=50,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        rng = np.random.default_rng(
            self.random_state
        )

        self.trees = []

        n_samples = len(X)


        for _ in range(self.n_estimators):

            # Bootstrap
            indices = rng.integers(
                0,
                n_samples,
                size=n_samples
            )

            X_sample = X[indices]
            y_sample = y[indices]


            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.integers(
                    0,
                    1_000_000
                )
            )


            tree.fit(
                X_sample,
                y_sample
            )

            self.trees.append(tree)


        return self


    def predict(self, X):
        predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        # Голосование деревьев
        return (
            predictions.mean(axis=0) >= 0.5
        ).astype(int)