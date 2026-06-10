"""
Serialize final dataframe to Parquet and write manifest.
"""
import os
import json
from datetime import datetime
from agents.utils.logger_setup import get_logger

logger = get_logger('serializer')


def write_parquet_and_manifest(df, out_dir: str, feature_order: list):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'offers_observations.parquet')
    try:
        df.write_parquet(out_path)
        logger.info(f'Wrote features to {out_path}')
    except Exception:
        logger.exception('Failed to write parquet')
        raise
    manifest = {
        'rows': df.height,
        'feature_order': feature_order,
        'created': datetime.utcnow().isoformat() + 'Z'
    }
    try:
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info('Manifest written')
    except Exception:
        logger.exception('Failed to write manifest')
        raise
