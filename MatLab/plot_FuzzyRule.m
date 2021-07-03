%%
ch_folder = 'csvs/';
tmp = dir([ch_folder '*Predict_Stock_0941_FuzzyRule.csv']);
for i = 1:length(tmp)
    tmp_ch = sprintf('%s%s', ch_folder, tmp(i).name);
    tmp_dt = importdata(tmp_ch);
    subchs = strsplit(tmp_ch, {'_', '/'});
    tmp_tt = strrep(subchs{2}, 'evo', '');
    disp(tmp_tt)
    disp(mean(tmp_dt(1,:)));
    %createfigure_FuzzyRule(tmp_dt', tmp_tt);
    %fname = sprintf('FIGUREs/fuzzyrule%s.eps',tmp_tt);
    %print(fname,'-depsc2','-r300');
    %fname = sprintf('FIGUREs/fuzzyrule%s.fig',tmp_tt);
    %savefig(fname);
end

