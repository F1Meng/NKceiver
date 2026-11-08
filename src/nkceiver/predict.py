from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse


def _require_fields(mapping, fields, source):
    if not isinstance(mapping, dict):
        raise ValueError(f"{source} must be a dictionary")
    missing = sorted(set(fields) - mapping.keys())
    if missing:
        raise ValueError(f"{source} is missing fields: {', '.join(missing)}")


def _check_names(names, field):
    if not isinstance(names, list) or not names:
        raise ValueError(f"{field} must be a nonempty list")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{field} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must contain unique names")


def _check_array(value, shape, field):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain finite numbers")
    return array


def _single_checkpoint(checkpoints):
    if not isinstance(checkpoints, (list, tuple)) or len(checkpoints) != 1:
        raise ValueError("checkpoints must contain exactly one checkpoint")
    if not isinstance(checkpoints[0], (str, Path)) or not str(checkpoints[0]):
        raise ValueError("checkpoints[0] must be a checkpoint filename")
    return checkpoints[0]


def _validate_checkpoint(checkpoint):
    _require_fields(
        checkpoint,
        ["format_version", "model", "config", "class_names", "mean", "std", "reference_M"],
        "checkpoint",
    )
    if checkpoint["format_version"] != 3:
        raise ValueError("checkpoint.format_version must be 3")
    _check_names(checkpoint["class_names"], "checkpoint.class_names")
    if len(checkpoint["class_names"]) < 2:
        raise ValueError("checkpoint.class_names must contain at least two classes")
    config = checkpoint["config"]
    _require_fields(
        config, ["architecture", "feature", "presence_mode", "dropout", "seed"], "checkpoint.config"
    )
    n_states = config.get("n_states", 12)
    if type(n_states) is not int or n_states < 1:
        raise ValueError("checkpoint.config.n_states must be a positive integer")
    n_genes = np.size(checkpoint["mean"])
    if n_genes < 1:
        raise ValueError("checkpoint.mean must contain gene normalization values")
    _check_array(checkpoint["mean"], (n_genes,), "checkpoint.mean")
    std = _check_array(checkpoint["std"], (n_genes,), "checkpoint.std")
    if (std < 0).any():
        raise ValueError("checkpoint.std must be nonnegative")
    _check_array(checkpoint["reference_M"], (n_states, n_genes), "checkpoint.reference_M")


def _validate_annotator(annotator, config):
    fields = [
        "format",
        "genes",
        "state_names",
        "expression_mean",
        "expression_std",
        "latent_mean",
        "latent_std",
        "anchor_latent",
        "anchor_labels",
        "mapper",
    ]
    _require_fields(annotator, fields, "annotator")
    if annotator["format"] != "frozen_atlas_mapping_v1":
        raise ValueError("annotator.format must be frozen_atlas_mapping_v1")
    for field in ["genes", "state_names"]:
        if annotator[field] != config[field]:
            raise ValueError(f"annotator.{field} must match bundle.{field} in order")
    n_genes = len(config["genes"])
    _check_array(annotator["expression_mean"], (n_genes,), "annotator.expression_mean")
    std = _check_array(annotator["expression_std"], (n_genes,), "annotator.expression_std")
    if (std <= 0).any():
        raise ValueError("annotator.expression_std must be positive")
    shape = np.shape(annotator["anchor_latent"])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("annotator.anchor_latent must be a nonempty matrix")
    n_anchors, dimensions = shape
    _check_array(annotator["anchor_latent"], shape, "annotator.anchor_latent")
    _check_array(annotator["latent_mean"], (dimensions,), "annotator.latent_mean")
    std = _check_array(annotator["latent_std"], (dimensions,), "annotator.latent_std")
    if (std <= 0).any():
        raise ValueError("annotator.latent_std must be positive")
    labels = _check_array(annotator["anchor_labels"], (n_anchors,), "annotator.anchor_labels")
    if (
        (labels < 0).any()
        or (labels >= len(config["state_names"])).any()
        or not np.equal(labels, np.rint(labels)).all()
    ):
        raise ValueError("annotator.anchor_labels must index bundle.state_names")
    neighbors = annotator.get("neighbors", 15)
    if type(neighbors) is not int or not 1 <= neighbors <= n_anchors:
        raise ValueError("annotator.neighbors must be between 1 and the anchor count")
    _require_fields(annotator["mapper"], ["0.weight", "5.weight"], "annotator.mapper")
    _check_array(annotator["mapper"]["0.weight"], (256, n_genes), "annotator.mapper.0.weight")
    _check_array(annotator["mapper"]["5.weight"], (dimensions, 128), "annotator.mapper.5.weight")


def load_bundle(bundle, task):
    """Load one task's mapping resources and classifier, validating their schema."""
    bundle = Path(bundle)
    config = json.loads((bundle / task / "bundle.json").read_text())
    _require_fields(
        config,
        [
            "format_version",
            "task",
            "genes",
            "state_names",
            "class_names",
            "checkpoints",
            "min_state_cells",
        ],
        "bundle",
    )
    if config["format_version"] != 3:
        raise ValueError("bundle.format_version must be 3")
    task_names = {"disease": ("disease",), "covid": ("covid", "covid2"), "ici": ("ici",)}
    # Existing COVID-19 bundles identify their binary task as covid2.
    if config["task"] not in task_names.get(task, ()):
        raise ValueError("bundle.task must match the selected task")
    for field in ["genes", "state_names", "class_names"]:
        _check_names(config[field], "bundle." + field)
    checkpoint_name = _single_checkpoint(config["checkpoints"])
    if type(config["min_state_cells"]) is not int or config["min_state_cells"] < 1:
        raise ValueError("bundle.min_state_cells must be a positive integer")
    model, checkpoint = load_classifier(bundle / task / checkpoint_name)
    if config["class_names"] != checkpoint["class_names"]:
        raise ValueError("bundle.class_names must match checkpoint.class_names in order")
    binary_classes = {
        "covid": ["non_severe", "severe_or_critical"],
        "ici": ["non_responder", "responder"],
    }
    if task in binary_classes and config["class_names"] != binary_classes[task]:
        raise ValueError(f"bundle.class_names for {task} must be {binary_classes[task]}")
    n_genes, n_states = len(config["genes"]), len(config["state_names"])
    _check_array(checkpoint["mean"], (n_genes,), "checkpoint.mean for bundle.genes")
    _check_array(
        checkpoint["reference_M"],
        (n_states, n_genes),
        "checkpoint.reference_M for bundle.state_names/genes",
    )
    if checkpoint["config"].get("n_states", 12) != n_states:
        raise ValueError("checkpoint.config.n_states must match bundle.state_names")
    minimum = checkpoint["config"].get("min_state_cells", config["min_state_cells"])
    if minimum != config["min_state_cells"]:
        raise ValueError("bundle.min_state_cells must match checkpoint.config.min_state_cells")
    annotator = torch.load(bundle / "annotator.pt", map_location="cpu", weights_only=False)
    _validate_annotator(annotator, config)
    return config, annotator, model, checkpoint


def checked_counts(x):
    x = sparse.csr_matrix(x, dtype=np.float32)
    if (
        not np.isfinite(x.data).all()
        or (x.data < 0).any()
        or (not np.equal(x.data, np.rint(x.data)).all())
    ):
        raise ValueError(
            "Expected finite, nonnegative integer raw counts; do not supply log-normalized expression"
        )
    return x


@torch.inference_mode()
def annotate_counts(counts, annotator, batch_size=512):
    if annotator.get("format") == "frozen_atlas_mapping_v1":
        from .mapping import transfer

        return transfer(counts, annotator, batch_size=batch_size)
    raise ValueError("Frozen atlas mapping weights required")


def aggregate_counts(counts, state, units, n_states=12):
    units = np.asarray(units, str)
    names, ids = np.unique(units, return_inverse=True)
    state = np.asarray(state, int)
    if (state < 0).any() or (state >= n_states).any():
        raise ValueError("State index outside annotation ontology")
    flat = ids * n_states + state
    grouping = sparse.csr_matrix(
        (np.ones(len(flat)), (flat, np.arange(len(flat)))), shape=(len(names) * n_states, len(flat))
    )
    sums = (
        grouping.dot(counts)
        .toarray()
        .astype(np.float32)
        .reshape(len(names), n_states, counts.shape[1])
    )
    n = np.bincount(flat, minlength=len(names) * n_states).reshape(len(names), n_states)
    p = (n / n.sum(1, keepdims=True)).astype(np.float32)
    library = sums.sum(-1, keepdims=True)
    phenotype = np.log1p(sums * (10000 / np.maximum(library, 1))).astype(np.float32)
    return (names, n, p, phenotype, sums)


def load_classifier(checkpoint):
    from .model import build_model

    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _validate_checkpoint(metadata)
    model = build_model(
        metadata["config"],
        torch.as_tensor(metadata["mean"]),
        torch.as_tensor(metadata["std"]),
        len(metadata["class_names"]),
        metadata["config"]["seed"],
    )
    try:
        model.load_state_dict(metadata["model"], strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError(
            "checkpoint.model parameters do not match its config and class_names"
        ) from error
    return model.eval(), metadata


@torch.inference_mode()
def _predict_model(model, n_classes, phenotype, proportions, present):
    usable = np.flatnonzero(present.any(1))
    result = np.full((len(phenotype), n_classes), np.nan, dtype=np.float32)
    for start in range(0, len(usable), 64):
        batch = usable[start : start + 64]
        logits = model(
            state_expression=torch.from_numpy(phenotype[batch]),
            state_proportion=torch.from_numpy(proportions[batch]),
            state_present=torch.from_numpy(present[batch]),
        ).leaf_logits
        result[batch] = logits.softmax(-1).numpy()
    return result


def predict_features(checkpoints, phenotype, proportions, present):
    model, metadata = load_classifier(_single_checkpoint(checkpoints))
    return _predict_model(model, len(metadata["class_names"]), phenotype, proportions, present)


def format_predictions(units, probability, names, task):
    result = pd.DataFrame({"sample_id": units})
    for i, name in enumerate(names):
        result["P::" + name] = probability[:, i]
    if task == "ici":
        responder = probability[:, names.index("responder")]
        result["responder_probability"] = responder
        result["predicted_class"] = np.where(responder >= 0.5, "responder", "non_responder")
        result["response_label"] = (responder >= 0.5).astype(int)
    else:
        predicted = (
            (probability[:, 1] >= 0.5).astype(int) if len(names) == 2 else probability.argmax(1)
        )
        result["predicted_class"] = np.asarray(names)[predicted]
        if task == "covid":
            result["severe_or_critical_probability"] = probability[
                :, names.index("severe_or_critical")
            ]
            result["severity_binary_label"] = (
                result["severe_or_critical_probability"] >= 0.5
            ).astype(int)
    valid = np.isfinite(probability).all(1)
    result["prediction_status"] = np.where(valid, "predicted", "insufficient_state_support")
    result.loc[~valid, "predicted_class"] = None
    for column in ("response_label", "severity_binary_label"):
        if column in result:
            result[column] = result[column].astype("Int64")
            result.loc[~valid, column] = pd.NA
    order = np.argsort(-probability, axis=1, kind="stable")
    if len(names) == 2:
        order[probability[:, 0] == probability[:, 1]] = [1, 0]
    result["prediction_ranking"] = [
        json.dumps(
            [
                {
                    "class": names[int(k)],
                    "probability": float(probability[i, k]),
                    "rank": int(j + 1),
                }
                for j, k in enumerate(row)
            ]
            if valid[i]
            else []
        )
        for i, row in enumerate(order)
    ]
    return result


def _prepare_counts(adata, config, sample_key, counts_key, patient_key):
    if adata.n_obs == 0:
        raise ValueError("Input must contain NK cells")
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Gene and cell identifiers must be unique")
    if sample_key not in adata.obs or adata.obs[sample_key].isna().any():
        raise ValueError(
            "A nonmissing sample identifier is required; labels and state annotations are not required"
        )
    if patient_key is not None:
        if patient_key not in adata.obs or adata.obs[patient_key].isna().any():
            raise ValueError("Patient identifier is missing")
        if adata.obs.groupby(sample_key, observed=True)[patient_key].nunique().max() != 1:
            raise ValueError("Each sample must belong to exactly one patient")
    positions = adata.var_names.get_indexer(config["genes"])
    if (positions < 0).any():
        raise ValueError(
            f"Missing {(positions < 0).sum()} required genes; unmeasured genes cannot silently be treated as zero"
        )
    raw = adata.X if counts_key == "X" else adata.layers[counts_key]
    counts = checked_counts(raw[:, positions])
    if (np.asarray(counts.sum(1)).ravel() == 0).any():
        raise ValueError(
            "Cells with zero counts on the fixed gene panel must be removed before prediction"
        )
    return counts


def _integrate_batch(model, checkpoint, phenotype, proportions, present, targets):
    from .attribution import integrated_state_contributions

    contributions = integrated_state_contributions(
        model, phenotype, proportions, present, checkpoint["reference_M"], targets
    )
    if np.max(np.abs(contributions["completeness_error"])) > 0.005:
        contributions = integrated_state_contributions(
            model, phenotype, proportions, present, checkpoint["reference_M"], targets, steps=256
        )
    for integration_steps in (1024, 4096):
        failed = np.flatnonzero(np.abs(contributions["completeness_error"]) > 0.005)
        if not len(failed):
            break
        ids = failed
        refined = integrated_state_contributions(
            model,
            phenotype[ids],
            proportions[ids],
            present[ids],
            checkpoint["reference_M"],
            targets[ids],
            steps=integration_steps,
        )
        for key in contributions:
            if isinstance(contributions[key], np.ndarray):
                contributions[key][failed] = refined[key]
    return contributions


def predict_h5ad(
    source, bundle, task, output, sample_key, counts_key="X", patient_key=None, explain=True
):
    import anndata as ad

    bundle, output = (Path(bundle), Path(output))
    config, annotator, model, checkpoint = load_bundle(bundle, task)
    state_names = config["state_names"]
    n_states = len(state_names)
    adata = ad.read_h5ad(source)
    counts = _prepare_counts(adata, config, sample_key, counts_key, patient_key)
    state, _ = annotate_counts(counts, annotator)
    units, n_cells, proportions, phenotype, _ = aggregate_counts(
        counts, state, adata.obs[sample_key].astype(str), n_states=n_states
    )
    valid = n_cells >= config.get("min_state_cells", 1)
    probability = _predict_model(model, len(config["class_names"]), phenotype, proportions, valid)
    predictions = format_predictions(units, probability, config["class_names"], task)
    predictions["n_NK_cells"] = n_cells.sum(1)
    predictions["below_training_minimum_50_cells"] = n_cells.sum(1) < 50
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    if patient_key is not None:
        sample_to_patient = (
            adata.obs.groupby(sample_key, observed=True)[patient_key].first().astype(str)
        )
        sample_to_patient.index = sample_to_patient.index.astype(str)
        probabilities = pd.DataFrame(probability, index=units, columns=config["class_names"])
        probabilities["patient_id"] = sample_to_patient.reindex(units).to_numpy()
        pooled = probabilities.groupby("patient_id")[config["class_names"]].mean()
        patient_predictions = format_predictions(
            pooled.index, pooled.to_numpy(), config["class_names"], task
        ).rename(columns={"sample_id": "patient_id"})
        patient_predictions.to_csv(output / "patient_predictions.csv", index=False)

    composition = pd.DataFrame(
        {
            "sample_id": np.repeat(units, n_states),
            "HNKA_L3": np.tile(state_names, len(units)),
            "state_index": np.tile(np.arange(n_states), len(units)),
            "n_cells": n_cells.ravel(),
            "proportion": proportions.ravel(),
            "presence": (n_cells > 0).ravel(),
            "valid_for_model": valid.ravel(),
            "min_state_cells": config.get("min_state_cells", 1),
        }
    )
    composition = composition.sort_values(
        ["sample_id", "proportion", "state_index"], ascending=[True, False, True]
    )
    composition["rank"] = composition.groupby("sample_id").cumcount() + 1
    composition.to_csv(output / "state_composition.csv", index=False)
    if patient_key is not None:
        c = composition.assign(patient_id=composition.sample_id.map(sample_to_patient))
        pc = c.groupby(["patient_id", "HNKA_L3", "state_index"], as_index=False).agg(
            n_cells=("n_cells", "sum"),
            model_mean_sample_proportion=("proportion", "mean"),
            presence=("presence", "max"),
        )
        pc["proportion"] = pc.n_cells / pc.groupby("patient_id").n_cells.transform("sum")
        pc = pc.sort_values(
            ["patient_id", "proportion", "state_index"], ascending=[True, False, True]
        )
        pc["rank"] = pc.groupby("patient_id").cumcount() + 1
        pc.to_csv(output / "patient_state_composition.csv", index=False)
    if explain and config.get("format_version") == 3:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        targets = probability.argmax(1) if task == "disease" else np.ones(len(units), dtype=int)
        if task == "disease" and patient_key is not None:
            target_map = pd.Series(pooled.to_numpy().argmax(1), index=pooled.index)
            targets = target_map.reindex(sample_to_patient.reindex(units).to_numpy()).to_numpy()
        records = []
        explain_ids = np.flatnonzero(np.isfinite(probability).all(1))
        if not len(explain_ids):
            return predictions
        for start in range(0, len(explain_ids), 8):
            batch = explain_ids[start : start + 8]
            contributions = _integrate_batch(
                model,
                checkpoint,
                phenotype[batch],
                proportions[batch],
                valid[batch],
                targets[batch],
            )
            for index, unit in enumerate(units[batch]):
                target_name = config["class_names"][targets[batch[index]]]
                for state_index in range(n_states):
                    records.append(
                        dict(
                            sample_id=unit,
                            HNKA_L3=state_names[state_index],
                            target_class=target_name,
                            P_contribution=float(contributions["P"][index, state_index]),
                            M_contribution=float(contributions["M"][index, state_index]),
                            total_contribution=float(
                                contributions["P"][index, state_index]
                                + contributions["M"][index, state_index]
                            ),
                            presence=bool(n_cells[batch[index], state_index] > 0),
                            valid_for_model=bool(valid[batch[index], state_index]),
                            reference_probability=float(
                                contributions["reference_probability"][index]
                            ),
                            target_probability=float(contributions["target_probability"][index]),
                            completeness_error=float(contributions["completeness_error"][index]),
                        )
                    )
        table = pd.DataFrame(records).sort_values(
            ["sample_id", "total_contribution"], ascending=[True, False]
        )
        table["signed_rank"] = table.groupby("sample_id").cumcount() + 1
        if checkpoint["config"]["feature"] == "M":
            table = table.drop(columns="P_contribution")
        table["attribution_quality_pass"] = table.completeness_error.abs() <= 0.005
        table.to_csv(output / "state_contributions.csv", index=False)
        if patient_key is not None:
            table["patient_id"] = table.sample_id.map(sample_to_patient)
            columns = [
                c
                for c in [
                    "P_contribution",
                    "M_contribution",
                    "total_contribution",
                    "reference_probability",
                    "target_probability",
                    "completeness_error",
                ]
                if c in table
            ]
            pt = table.groupby(["patient_id", "HNKA_L3", "target_class"], as_index=False)[
                columns
            ].mean()
            pt = pt.sort_values(["patient_id", "total_contribution"], ascending=[True, False])
            pt["signed_rank"] = pt.groupby("patient_id").cumcount() + 1
            quality = (
                table.groupby(["patient_id", "HNKA_L3", "target_class"])
                .completeness_error.agg(lambda x: float(x.abs().max()))
                .rename("max_sample_completeness_error")
                .reset_index()
            )
            pt = pt.merge(
                quality, on=["patient_id", "HNKA_L3", "target_class"], validate="one_to_one"
            )
            pt["attribution_quality_pass"] = pt.max_sample_completeness_error <= 0.005
            pt.to_csv(output / "patient_state_contributions.csv", index=False)
        (output / "attribution_method.json").write_text(
            json.dumps(
                {
                    "model_input": checkpoint["config"]["feature"],
                    "method": "integrated gradients of target probability with respect to state expression, fixed validity mask",
                    "reference": "training-only state phenotype means restricted to valid states; composition is not adata predictive input",
                    "absence_contribution": "not attributed; presence held fixed",
                    "integration_steps": "64, retry 256 then refine failed samples contributions 1024/4096; quality flag requires residual <= 0.005",
                },
                indent=2,
            )
        )
    return predictions


def _write_report(output, donor, task, prediction, states, top_k):
    shown = states[states.n_cells > 0]
    shown = shown if top_k is None else shown.head(top_k)
    lines = [
        f"# Donor prediction: {donor}",
        f"Task: {task}. Predicted class: {prediction.predicted_class.iloc[0]}. Status: {prediction.prediction_status.iloc[0]}.",
        f"Eligible NK cells: {int(states.n_cells.sum())}. States observed: {int((states.n_cells > 0).sum())}; states used by at least one sample: {int(states.used_in_model.sum())}.",
        "",
        "| Proportion rank | State | Cells | Proportion | Used | Signed contribution | Contribution rank |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in shown.itertuples():
        lines.append(
            f"| {row.proportion_rank} | {row.state} | {row.n_cells} | {row.proportion:.2%} | {row.used_in_model} | "
            + (
                f"{row.signed_contribution:+.5f} | {row.contribution_rank} |"
                if pd.notna(row.signed_contribution)
                else "not attributed | — |"
            )
        )
    attributed = states[states.used_in_model & states.signed_contribution.notna()]
    if len(attributed):
        row = attributed.iloc[0]
        lines += [
            "",
            f"Explanation target: {row.explanation_target}. Target probability: {row.target_probability:.6f}; training-reference probability: {row.reference_probability:.6f}.",
            f"Integrated-gradient numerical quality passed: {bool(attributed.attribution_quality_pass.all())}.",
        ]
    lines += [
        "",
        (
            "Proportions use all eligible input NK cells, including cells in low-support states. "
            "Proportion rank and contribution rank are computed independently. "
            "Contribution rank sorts signed values from positive to negative; "
            "absolute_contribution_rank separately measures magnitude."
        ),
        (
            "Signed contributions are integrated gradients relative to the training reference, "
            "with the state-support mask held fixed. Contributions are reported for supported states."
        ),
        (
            "For multiple original samples from this donor, prediction and contributions are "
            "averaged over usable samples; proportions pool all eligible cells. "
            "The two summaries therefore use different, explicitly defined weighting."
        ),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def predict_donor(
    source, bundle, task, output, sample_key=None, patient_key=None, counts_key="X", top_k=None
):
    import tempfile
    import anndata as ad

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="nkceiver-donor-") as temp:
        temp = Path(temp)
        adata = ad.read_h5ad(source)
        if patient_key is not None:
            if (
                patient_key not in adata.obs
                or adata.obs[patient_key].isna().any()
                or adata.obs[patient_key].astype(str).nunique() != 1
            ):
                raise ValueError("Supply exactly one donor for the compact donor interface")
            donor = str(adata.obs[patient_key].iloc[0])
        else:
            donor = "input_donor"
        if sample_key is None:
            adata.obs["__nkceiver_sample"] = "sample"
            sample_key = "__nkceiver_sample"
        adata.obs["__nkceiver_donor"] = donor
        adata.write_h5ad(temp / "input.h5ad")
        predict_h5ad(
            temp / "input.h5ad",
            bundle,
            task,
            temp / "detailed",
            sample_key,
            counts_key,
            "__nkceiver_donor",
            True,
        )
        details = temp / "detailed"
        prediction = pd.read_csv(details / "patient_predictions.csv")
        prediction.insert(1, "task", task)
        keep = (
            ["patient_id", "task", "predicted_class"]
            + [c for c in prediction if c.startswith("P::")]
            + ["prediction_status"]
        )
        prediction = prediction[keep].rename(columns={"patient_id": "donor_id"})
        composition = pd.read_csv(details / "patient_state_composition.csv")
        sample_composition = pd.read_csv(details / "state_composition.csv")
        valid = (
            sample_composition.groupby("HNKA_L3")
            .valid_for_model.agg(["any", "mean"])
            .rename(columns={"any": "used_in_model", "mean": "valid_sample_fraction"})
        )
        states = (
            composition[["HNKA_L3", "state_index", "n_cells", "proportion", "rank"]]
            .rename(columns={"HNKA_L3": "state", "rank": "proportion_rank"})
            .merge(valid, left_on="state", right_index=True, validate="one_to_one")
        )
        contribution_path = details / "patient_state_contributions.csv"
        if contribution_path.exists():
            contributions = pd.read_csv(contribution_path).rename(
                columns={
                    "HNKA_L3": "state",
                    "target_class": "explanation_target",
                    "total_contribution": "signed_contribution",
                }
            )
            columns = [
                "state",
                "explanation_target",
                "signed_contribution",
                "reference_probability",
                "target_probability",
                "attribution_quality_pass",
                "max_sample_completeness_error",
            ]
            states = states.merge(
                contributions[columns], on="state", how="left", validate="one_to_one"
            )
        else:
            for column in (
                "signed_contribution",
                "reference_probability",
                "target_probability",
                "max_sample_completeness_error",
            ):
                states[column] = np.nan
            states["explanation_target"] = {"ici": "responder", "covid": "severe_or_critical"}.get(
                task
            )
            states["attribution_quality_pass"] = False
        states.loc[~states.used_in_model, "signed_contribution"] = np.nan
        states["positive_contribution"] = states.signed_contribution.clip(lower=0)
        states["negative_contribution"] = states.signed_contribution.clip(upper=0)
        states["contribution_rank"] = states.signed_contribution.rank(
            ascending=False, method="first"
        ).astype("Int64")
        states["absolute_contribution_rank"] = (
            states.signed_contribution.abs().rank(ascending=False, method="first").astype("Int64")
        )
        states["model_status"] = np.where(
            states.used_in_model, "observed_expression_used", "excluded_by_attention_mask"
        )
        states.insert(0, "donor_id", donor)
        states = states.sort_values(["proportion_rank", "state_index"]).drop(columns="state_index")
        output.mkdir(parents=True)
        prediction.to_csv(output / "prediction.csv", index=False)
        states.to_csv(output / "states.csv", index=False)
        _write_report(output, donor, task, prediction, states, top_k)
    return (prediction, states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", choices=["disease", "covid", "ici"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-key")
    parser.add_argument(
        "--patient-key", help="Optional patient ID for equal averaging of sample probabilities"
    )
    parser.add_argument("--counts-key", default="X", help="X or name of the raw-count layer")
    parser.add_argument(
        "--top-k",
        type=int,
        help="Limit observed-state rows in the report only; the state table remains complete",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    predict_donor(
        args.input,
        args.bundle,
        args.task,
        args.output,
        args.sample_key,
        args.patient_key,
        args.counts_key,
        args.top_k,
    )


if __name__ == "__main__":
    main()
