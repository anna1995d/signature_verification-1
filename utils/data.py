from data import Data
from utils.config import CONFIG


DATA = Data(
    smp_stp=CONFIG.smp_stp,
    rl_win_sz=CONFIG.rl_win_sz,
    rl_win_stp=CONFIG.rl_win_stp,
    ftr_cnt=CONFIG.ftr_cnt,
    nrm=CONFIG.nrm,
    usr_cnt=CONFIG.usr_cnt,
    gen_smp_cnt=CONFIG.gen_smp_cnt,
    frg_smp_cnt=CONFIG.frg_smp_cnt,
    gen_path_temp=CONFIG.gen_path_temp,
    frg_path_temp=CONFIG.frg_path_temp
)
