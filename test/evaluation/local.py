from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/zx/datasets/got10k_lmdb'
    settings.got10k_path = '/zx/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/zx/datasets/itb'
    settings.lasot_extension_subset_path_path = '/zx/datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/zx/datasets/lasot_lmdb'
    settings.lasot_path = '/zx/datasets/lasot'
    settings.network_path = '/zx/projects/FGTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/zx/datasets/nfs'
    settings.otb_path = '/zx/datasets/otb'
    settings.prj_dir = '/zx/projects/FGTrack'
    settings.result_plot_path = '/zx/projects/FGTrack/output/test/result_plots'
    settings.results_path = '/zx/projects/FGTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/zx/projects/FGTrack/output'
    settings.segmentation_path = '/zx/projects/FGTrack/output/test/segmentation_results'
    settings.tc128_path = '/zx/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/zx/datasets/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/zx/datasets/trackingnet'
    settings.uav_path = '/zx/datasets/uav'
    settings.vot18_path = '/zx/datasets/vot2018'
    settings.vot22_path = '/zx/datasets/vot2022'
    settings.vot_path = '/zx/datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

