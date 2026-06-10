"""
Orchestrator CLI to run Step A pipeline in modular steps.
"""
import argparse
from agents.utils.logger_setup import get_logger
from .loader import load_main_csv, load_cycles_csv
from .filter_prebook import filter_prebooked_waybills
from .explode_cycles import explode_cycles
from .join_main_info import join_main_info
from .history_builder import build_courier_day_history
from .active_seconds import compute_active_seconds
from .features_assembler import assemble_features
from .serializer import write_parquet_and_manifest

logger = get_logger('cli')


def run(full_args):
    main_df = load_main_csv(full_args.main)
    cycles_df = load_cycles_csv(full_args.cycles)
    cycles_clean, removed = filter_prebooked_waybills(main_df, cycles_df)
    offers_df = explode_cycles(cycles_clean)
    offers_df = join_main_info(offers_df, main_df)
    offers_df = build_courier_day_history(offers_df)
    offers_df, batches = compute_active_seconds(main_df, offers_df)
    out_df = assemble_features(offers_df)
    feature_order = [
        'sender_lng', 'sender_lat', 'proxy_lng', 'proxy_lat', 'recipient_lng', 'recipient_lat',
        'tod_sin', 'tod_cos', 'capacity_at_dispatch', 
        'offers_so_far', 'accepts_so_far', 'completed_so_far', 'late_ratio_so_far', 'time_since_last_accept_s',
        'courier_active_seconds_so_far', 'eta_seconds_current', 'order_income_value'
    ]
    write_parquet_and_manifest(out_df, full_args.out_dir, feature_order)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--cycles', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    run(args)
