import numpy as np
import pandas as pd
from utils import *

class IncrementalRuleBasedClassifier:

    def __init__(self, min_support=5, coverage_threshold=0.5, improvement_threshold=0.05):

        self.T = None
        self.R = None
        self.A = None
        self.default_class = None
        self.classes = None  
        self.mapping = None

        self.min_support = min_support
        self.coverage_threshold = coverage_threshold
        self.improvement_threshold = improvement_threshold
        self.rules = []         
           
    def fit(self, X, y):
        
        self.mapping = X.columns 
        
        columns_mapping(df)
        X0, X1, X2, y = split_dataframe(df)
        X = np.concatenate((X0, X1, X0), axis=1)



    def create_rule(self, X, y):
        
        pass


    def predict(self, X):

        pass

    def score(self, X, y):

        pass



class WeightedIncrementalRuleBasedClassifier:
    def __init__(self, min_support=5, coverage_threshold=0.5, improvement_threshold=0.05, class_weights=None):
        """
        Parameters:
            min_support: Minimum number of records that a candidate rule must cover.
            coverage_threshold: For each class, stop generating rules when at least 
                                (coverage_threshold * 100)% of its training records have been covered.
            improvement_threshold: Minimum reduction in error (1 - fraction(target))
                                   required to add a new condition to the rule.
            class_weights: A dictionary mapping each class to its weight. 
                           Example: {0: 1.0, 1: 2.0, 2: 1.5}. If None, equal weights are assumed.
        """
        self.min_support = min_support
        self.coverage_threshold = coverage_threshold
        self.improvement_threshold = improvement_threshold
        self.class_weights = class_weights
        self.rules = []         # Will hold tuples of (rule, predicted_class)
        self.default_class = None
        self.classes = None     # List of classes encountered during training

    def fit(self, X, y):
        """
        Generate rules incrementally from the training data.
        
        Parameters:
            X: pandas DataFrame of features.
            y: pandas Series or array-like target labels.
        """
        # Combine features and target for easier processing
        data = X.copy()
        data['target'] = y

        # Order classes by relevance (here: frequency)
        class_counts = data['target'].value_counts()
        ordered_classes = class_counts.index.tolist()
        self.classes = ordered_classes

        # If no class weights provided, assign equal weight (1.0) to each class.
        if self.class_weights is None:
            self.class_weights = {cls: 1.0 for cls in ordered_classes}

        R = []         # The set of generated rules
        T = data.copy()  # Working copy of training records

        # For each class (in order of relevance)
        for current_class in ordered_classes:
            original_count = class_counts[current_class]
            # Continue generating rules until enough records for current_class are covered.
            while T[T['target'] == current_class].shape[0] > (1 - self.coverage_threshold) * original_count:
                # Generate a rule for current_class using the general-to-specific approach.
                rule = self._generate_rule_for_class(T, current_class, X.columns)
                if rule is None:
                    break  # No further improvement possible for this class.
                # Add the rule (with its associated class) to the rule set.
                R.append((rule, current_class))
                # Remove records covered by this rule from T.
                T = T[~T.apply(lambda row: self._rule_matches(rule, row), axis=1)]
        self.rules = R
        self.default_class = ordered_classes[0]
        return self

    def _generate_rule_for_class(self, T, target_class, features):
        """
        Generates a candidate rule for target_class using a general-to-specific approach.
        
        The process starts with an empty rule (covering all records) and iteratively adds one 
        condition at a time. At each step, the candidate condition that yields the best improvement 
        (i.e., reduction in error defined as 1 - fraction of records that are of target_class) is selected.
        The procedure stops when no candidate condition provides an improvement of at least 
        'improvement_threshold'.
        
        Parameters:
            T: DataFrame containing current training records (including 'target').
            target_class: The class for which to generate a rule.
            features: List of feature names.
            
        Returns:
            A rule as a list of conditions [(feature, operator, threshold), ...] or None if no rule is found.
        """
        # Start with an empty rule (i.e., no conditions).
        current_rule = []
        # Initially, the rule covers all records in T.
        current_covered = T.copy()
        current_error = self._error(current_covered, target_class)
        
        while True:
            best_candidate = None
            best_candidate_error = current_error
            # Iterate over all features and all candidate thresholds in the current covered set.
            for feature in features:
                unique_values = sorted(current_covered[feature].unique())
                if len(unique_values) < 2:
                    continue  # Skip if there is not enough variability.
                for threshold in unique_values:
                    for op in ['<=', '>']:
                        candidate_condition = (feature, op, threshold)
                        # Create a candidate rule by adding this condition.
                        candidate_rule = current_rule + [candidate_condition]
                        # Filter current_covered further using only the new condition.
                        if op == '<=':
                            candidate_covered = current_covered[current_covered[feature] <= threshold]
                        else:
                            candidate_covered = current_covered[current_covered[feature] > threshold]
                        if candidate_covered.shape[0] < self.min_support:
                            continue
                        candidate_error = self._error(candidate_covered, target_class)
                        # Check if this condition yields a better (lower) error.
                        if candidate_error < best_candidate_error:
                            best_candidate_error = candidate_error
                            best_candidate = candidate_condition
            # If a candidate condition was found that improves error by at least the threshold, add it.
            if best_candidate is not None and (current_error - best_candidate_error) >= self.improvement_threshold:
                current_rule.append(best_candidate)
                feature, op, threshold = best_candidate
                if op == '<=':
                    current_covered = current_covered[current_covered[feature] <= threshold]
                else:
                    current_covered = current_covered[current_covered[feature] > threshold]
                current_error = best_candidate_error
            else:
                break  # No candidate condition meets the improvement criterion.
        
        # Return the rule if at least one condition was added.
        return current_rule if current_rule else None

    def _error(self, data, target_class):
        """
        Compute the error rate for the candidate rule with respect to target_class.
        
        Error is defined as 1 - (fraction of records in data that belong to target_class).
        If no records are covered, return 1.0.
        """
        if data.shape[0] == 0:
            return 1.0
        p = (data['target'] == target_class).mean()
        return 1 - p

    def _rule_matches(self, rule, row):
        """
        Check if a record (row) satisfies all conditions in a rule.
        
        Parameters:
            rule: A list of conditions (each condition is a tuple: (feature, operator, threshold)).
            row: A pandas Series representing a record.
            
        Returns:
            True if the row meets all conditions; False otherwise.
        """
        for feature, op, threshold in rule:
            if op == '<=':
                if not row[feature] <= threshold:
                    return False
            elif op == '>':
                if not row[feature] > threshold:
                    return False
        return True

    def predict(self, X):
        """
        Classify each record in X using a weighted vote mechanism.
        
        For each test record:
          - Initialize a vote mapping F (one entry per class, starting at 0).
          - For every rule in the generated rule set, if the rule covers the record,
            add the pre-defined weight (from class_weights) for that rule's predicted class to F.
          - Output the class with the highest vote. If no rule fires, use the default class.
        
        Parameters:
            X: pandas DataFrame of features.
            
        Returns:
            A numpy array of predicted class labels.
        """
        predictions = []
        for _, row in X.iterrows():
            votes = {cls: 0 for cls in self.classes}
            for rule, predicted_class in self.rules:
                if self._rule_matches(rule, row):
                    votes[predicted_class] += self.class_weights.get(predicted_class, 1.0)
            if all(vote == 0 for vote in votes.values()):
                predictions.append(self.default_class)
            else:
                predicted = max(votes, key=votes.get)
                predictions.append(predicted)
        return np.array(predictions)

    def print_rules(self):
        """
        Print the generated rules in a human-readable format.
        """
        print("Generated Rules (unordered):")
        for idx, (rule, predicted_class) in enumerate(self.rules, 1):
            conditions = " AND ".join([f"{feat} {op} {threshold:.2f}" for feat, op, threshold in rule])
            print(f"Rule {idx}: IF {conditions} THEN class = {predicted_class}")
        print(f"Default rule: ELSE predict class = {self.default_class}")


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Load sample data (Iris dataset for demonstration)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['target']),
        df['target'],
        test_size=0.3,
        random_state=42,
        stratify=df['target']
    )

    # Optionally, define class weights (e.g., giving more weight to a specific class)
    weights = {0: 1.0, 1: 2.0, 2: 1.5}

    # Create and train the classifier
    classifier = WeightedIncrementalRuleBasedClassifier(
        min_support=5,
        coverage_threshold=0.5,
        improvement_threshold=0.05,
        class_weights=weights
    )
    classifier.fit(X_train, y_train)

    # Print the generated rules
    classifier.print_rules()

    # Predict on the test set using weighted voting
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
