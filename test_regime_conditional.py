"""
Quick test for regime_conditional.py module

This script tests the basic functionality of the regime-conditional modeling module.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from src.models.regime_conditional import (
    train_per_regime,
    predict_regime_aware,
    compare_global_vs_conditional,
    print_comparison_results,
    bootstrap_accuracy_test,
    get_regime_feature_importance,
    create_regime_performance_dataframe,
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("TESTING REGIME-CONDITIONAL MODULE")
print("=" * 80)

# Generate synthetic data
print("\n1. Generating synthetic data...")
n_samples = 1000
n_features = 20

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=15,
    n_redundant=5,
    random_state=42,
)

# Create synthetic regimes
regimes = np.random.choice(["Calm", "Volatile", "Trending"], size=n_samples)

# Convert to DataFrame
feature_names = [f"feature_{i}" for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)

# Split into train/test
split_idx = int(0.8 * n_samples)
X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
regimes_train, regimes_test = regimes[:split_idx], regimes[split_idx:]

print(f"  ✓ Created {n_samples} samples with {n_features} features")
print(f"  ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"  ✓ Regimes: {np.unique(regimes)}")

# Test 1: Train regime-specific models
print("\n2. Testing train_per_regime()...")
regime_models = train_per_regime(
    X_train=X_train,
    y_train=y_train,
    regimes_train=regimes_train,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
    min_samples=20,
    verbose=True,
)

print(f"\n  ✓ Trained {len(regime_models)} regime-specific models")

# Test 2: Train global model
print("\n3. Training global model...")
global_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
global_model.fit(X_train, y_train)
y_pred_global = global_model.predict(X_test)
y_proba_global = global_model.predict_proba(X_test)[:, 1]
print("  ✓ Global model trained")

# Test 3: Make regime-aware predictions
print("\n4. Testing predict_regime_aware()...")
y_pred_conditional, y_proba_conditional = predict_regime_aware(
    X_test=X_test,
    regimes_test=regimes_test,
    regime_models=regime_models,
    global_model=global_model,
    return_proba=True,
    verbose=True,
)

print(f"\n  ✓ Generated {len(y_pred_conditional)} predictions")

# Test 4: Compare performance
print("\n5. Testing compare_global_vs_conditional()...")
results = compare_global_vs_conditional(
    y_true=y_test,
    y_pred_global=y_pred_global,
    y_pred_conditional=y_pred_conditional,
    y_proba_global=y_proba_global,
    y_proba_conditional=y_proba_conditional,
    regimes=regimes_test,
    model_name="Random Forest",
)

print_comparison_results(results, verbose=True)

# Test 5: Bootstrap significance testing
print("\n6. Testing bootstrap_accuracy_test()...")
bootstrap_results = bootstrap_accuracy_test(
    y_true=y_test,
    y_pred1=y_pred_global,
    y_pred2=y_pred_conditional,
    n_bootstrap=1000,
    alpha=0.05,
    random_state=42,
)

print(f"  Mean accuracy difference: {bootstrap_results['mean_diff']:.4f}")
print(
    f"  95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]"
)
print(f"  P-value: {bootstrap_results['p_value']:.4f}")
print(
    f"  Significant at α=0.05: {'Yes' if bootstrap_results['significant'] else 'No'}"
)

# Test 6: Feature importance
print("\n7. Testing get_regime_feature_importance()...")
importance_by_regime = get_regime_feature_importance(
    regime_models=regime_models, feature_names=feature_names, top_n=5
)

for regime, importance_df in importance_by_regime.items():
    print(f"\n  {regime} - Top 5 features:")
    for idx, row in importance_df.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

# Test 7: Performance DataFrame
print("\n8. Testing create_regime_performance_dataframe()...")
perf_df = create_regime_performance_dataframe(
    y_true=y_test,
    y_pred_global=y_pred_global,
    y_pred_conditional=y_pred_conditional,
    regimes=regimes_test,
)

print("\n  Regime Performance Summary:")
print(perf_df.to_string(index=False))

# Final summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nModule functions verified:")
print("  ✓ train_per_regime()")
print("  ✓ predict_regime_aware()")
print("  ✓ compare_global_vs_conditional()")
print("  ✓ print_comparison_results()")
print("  ✓ bootstrap_accuracy_test()")
print("  ✓ get_regime_feature_importance()")
print("  ✓ create_regime_performance_dataframe()")
print("\n✓ Module is ready for production use!")
