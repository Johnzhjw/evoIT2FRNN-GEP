function []=TIME()
%% 
        times=zeros(12,2);
        
        tmp=importdata('DPCCMOLSEA-NeuroEvo-ori/OUTPUT/TIME_DPCCMOLSEA_1546003029.csv');
        MEANTIMEDPori=tmp.data;
        tmp=importdata('DPCCMOLSEA-NeuroEvo-xwt/OUTPUT/TIME_DPCCMOLSEA_1546037160.csv');
        MEANTIMEDPxwt=tmp.data;

        times(:,1)=MEANTIMEDPori(:,end);
        times(:,2)=MEANTIMEDPxwt(:,end);

        save('TIME_ALL.mat','times');

%close all

