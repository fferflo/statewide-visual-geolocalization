import jax
import einx
import jax.numpy as jnp
import tiledwebmaps as twm

def decoupled_log_softmax(x, axis, where, eye):
    x = jnp.where(where, x, 0.0)

    # Nominator
    pos_x = x

    # Denominator with positive term
    neg_x = x
    neg_x_max = jax.lax.stop_gradient(jnp.max(neg_x, axis, where=where, initial=-jnp.inf, keepdims=True))
    neg_x = jnp.log(jnp.sum(jnp.exp(neg_x - neg_x_max), axis, where=where, keepdims=True)) + neg_x_max

    # Denominator without positive term
    x_max = jax.lax.stop_gradient(jnp.maximum(pos_x, neg_x))
    neg_x = jnp.exp(neg_x - x_max) - jnp.exp(pos_x - x_max)
    neg_x = jnp.maximum(neg_x, 1e-40)
    neg_x = jnp.log(neg_x) + x_max

    return jnp.where(where, pos_x - neg_x, 0.0)

def log_softmax(x, axis, where, eye):
    x = jax.nn.log_softmax(x, axis=axis, where=where, initial=-jnp.inf)
    x = jnp.where(where, x, 0.0)
    return x

def crossentropy(batch, model_output, min_distance, eps, decoupled):
    assert isinstance(decoupled, bool)
    metrics = {}

    # Gather embeddings from all devices
    pv_feature_all     = jax.lax.all_gather(model_output.pv_features, axis_name="devices", axis=0, tiled=True)
    aerial_feature_all = jax.lax.all_gather(model_output.aerial_features, axis_name="devices", axis=0, tiled=True)
    pv_latlons_all     = jax.lax.all_gather(batch.pv.latlons, axis_name="devices", axis=0, tiled=True)
    aerial_latlons_all = jax.lax.all_gather(batch.aerial.latlons, axis_name="devices", axis=0, tiled=True)

    # Find valid queries, references and pairs
    eye = einx.equal("bg, ba -> bg ba", jnp.arange(pv_feature_all.shape[0]), jnp.arange(aerial_feature_all.shape[0]))

    # with jax.experimental.enable_x64():
    distances_meter = twm.geo.distance(pv_latlons_all[:, jnp.newaxis], aerial_latlons_all[jnp.newaxis, :], np=jnp) # bg ba
    mask = jnp.logical_or(distances_meter > min_distance, eye)

    num_references = einx.count_nonzero("bg [ba]", mask)
    num_queries    = einx.count_nonzero("[bg] ba", mask)
    valid_reference = num_queries >= 2
    valid_query     = num_references >= 2
    metrics["ref-per-qry"] = jnp.mean(num_references)
    metrics["qry-per-ref"] = jnp.mean(num_queries)

    # Compute pairwise similarities
    logits = einx.dot("bg c, ba c -> bg ba", pv_feature_all, aerial_feature_all)
    logits = jnp.where(mask, logits, -jnp.inf)

    # Compute batch statistics
    bs = jnp.arange(logits.shape[0])
    gt_logits = logits[bs, bs]
    above_gt = einx.greater("bg ba, bg", logits, gt_logits)
    above_gt = jnp.logical_and(above_gt, mask)
    num_above_gt = einx.count_nonzero("bg [ba]", above_gt)
    metrics["batch-r@1"] = jnp.sum(jnp.where(num_above_gt == 0, 1.0, 0.0)) / jnp.sum(jnp.where(valid_query, 1.0, 0.0))
    metrics["batch-median-k"] = jnp.median(num_above_gt)
    metrics["batch-mean-k"] = jnp.mean(num_above_gt)



    # Compute loss with query=pv, reference=aerial
    logits1 = (decoupled_log_softmax if decoupled else log_softmax)(logits, axis=1, where=mask, eye=eye)
    gt_probs1 = einx.where("bg ba, , bg", eye, 1.0 - eps, eps / jnp.maximum(num_references - 1, 1))
    loss1 = -einx.dot("bg ba, bg ba -> bg", logits1, gt_probs1)
    loss1 = jnp.where(valid_query, loss1, 0.0)
    loss1 = einx.mean("[bg]", loss1)

    # Compute loss with query=aerial, reference=pv
    logits2 = (decoupled_log_softmax if decoupled else log_softmax)(logits, axis=0, where=mask, eye=eye)
    gt_probs2 = einx.where("bg ba, , ba", eye, 1.0 - eps, eps / jnp.maximum(num_queries - 1, 1))
    loss2 = -einx.dot("bg ba, bg ba -> ba", logits2, gt_probs2)
    loss2 = jnp.where(valid_reference, loss2, 0.0)
    loss2 = einx.mean("[ba]", loss2)

    loss = 0.5 * (loss1 + loss2)
    metrics["loss"] = loss

    return loss, metrics



def triplet(batch, model_output, min_distance, safa_fix):
    # https://github.com/Jeff-Zilence/TransGeo2022/blob/main/criterion/soft_triplet.py
    # https://github.com/YujiaoShi/cross_view_localization_SAFA/blob/master/script/train_cvusa.py#L62
    metrics = {}

    # Gather embeddings from all devices
    pv_feature_all     = jax.lax.all_gather(model_output.pv_features, axis_name="devices", axis=0, tiled=True)
    aerial_feature_all = jax.lax.all_gather(model_output.aerial_features, axis_name="devices", axis=0, tiled=True)
    pv_latlons_all     = jax.lax.all_gather(batch.pv.latlons, axis_name="devices", axis=0, tiled=True)
    aerial_latlons_all = jax.lax.all_gather(batch.aerial.latlons, axis_name="devices", axis=0, tiled=True)

    # Find valid queries, references and pairs
    eye = einx.equal("bg, ba -> bg ba", jnp.arange(pv_feature_all.shape[0]), jnp.arange(aerial_feature_all.shape[0]))

    # with jax.experimental.enable_x64():
    distances_meter = twm.geo.distance(pv_latlons_all[:, jnp.newaxis], aerial_latlons_all[jnp.newaxis, :], np=jnp) # bg ba
    mask = jnp.logical_or(distances_meter > min_distance, eye)

    num_references = einx.count_nonzero("bg [ba]", mask)
    num_queries    = einx.count_nonzero("[bg] ba", mask)
    valid_reference = num_queries >= 2
    valid_query     = num_references >= 2
    metrics["ref-per-qry"] = jnp.mean(num_references)
    metrics["qry-per-ref"] = jnp.mean(num_queries)

    # Compute pairwise similarities
    logits = einx.dot("bg c, ba c -> bg ba", pv_feature_all, aerial_feature_all)
    logits = jnp.where(mask, logits, 0.0)
    assert logits.shape[0] == logits.shape[1]

    # Compute batch statistics
    bs = jnp.arange(logits.shape[0])
    gt_logits = logits[bs, bs]
    above_gt = einx.greater("bg ba, bg", logits, gt_logits)
    above_gt = jnp.logical_and(above_gt, mask)
    num_above_gt = einx.count_nonzero("bg [ba]", above_gt)
    metrics["batch-r@1"] = jnp.sum(jnp.where(num_above_gt == 0, 1.0, 0.0)) / jnp.sum(jnp.where(valid_query, 1.0, 0.0))
    metrics["batch-median-k"] = jnp.median(num_above_gt)
    metrics["batch-mean-k"] = jnp.mean(num_above_gt)



    if safa_fix:
        logits = -2 + 2 * logits # bg ba
    pos_logits = jnp.diag(logits) # b
    mask = jnp.logical_and(mask, jnp.logical_not(eye))
    pair_n = jnp.count_nonzero(mask)

    # Compute loss with query=aerial, reference=pv
    logits1 = einx.add("bg ba, ba", logits, -pos_logits)
    logits1 = jnp.where(mask, logits1, 0.0)
    loss1 = jnp.log(1 + jnp.exp(logits1))
    loss1 = jnp.where(mask, loss1, 0.0)
    loss1 = jnp.sum(loss1) / pair_n

    # Compute loss with query=pv, reference=aerial
    logits2 = einx.add("bg ba, bg", logits, -pos_logits)
    logits2 = jnp.where(mask, logits2, 0.0)
    loss2 = jnp.log(1 + jnp.exp(logits2))
    loss2 = jnp.where(mask, loss2, 0.0)
    loss2 = jnp.sum(loss2) / pair_n

    loss = 0.5 * (loss1 + loss2)
    metrics["loss"] = loss

    return loss, metrics
