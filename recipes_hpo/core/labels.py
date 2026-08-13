def label_names(feature):
    """Loaders declare labels as [ClassLabel], Sequence(ClassLabel) or ClassLabel."""
    if isinstance(feature, list):
        return list(feature[0].names)
    if hasattr(feature, "feature"):
        return list(feature.feature.names)
    return list(feature.names)
