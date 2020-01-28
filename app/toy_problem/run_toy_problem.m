% run_toy_problem - MATLAB script to load time-series data (wavelet
% coefficients resulting from compressed sensing of the transient response
% of the 2D heat equation on a square domain with randomly chosen holes)
% and subsequently utilize karma4 library for autoregressive model
% parameter calibration and order selection with forecasting
%
% Authors: Moe Khalil, Jina Lee, Maher Salloum
% Sandia National Labopratories
% email: mkhalil@sandia.gov
% Website: http://www.sandia.gov/~mnsallo/software.html
% August 2017; Last revision: 17-August-2017

% Cleaning environment
clear all; close all;

% reading data
matfile = ['data_toy.mat'];
% Change the path to the .mat file as necessary
load(matfile,'n_timeseries','wc_timeseries');

% assigning amounts of data to use and forecast
nstart = 1;
num_ts = 50;
num_forecast = 20;
nce = 60;
xv=0.3;
adap=1;

% sorting the data in order to start with the coeffs of larger amplitude
num_nz_coeff=(1:max(n_timeseries));
[w_cs, iw_cs] = sort(abs(wc_timeseries), 1, 'descend');
iw_cs=iw_cs(:,nstart);

% extract the chunk of data we need to send to the solver
Z=wc_timeseries(iw_cs(1:nce),nstart:nstart+num_ts+num_forecast-1);
Z=Z';
nz=size(Z);

% writing the data to disk
fid = fopen(['z.dat'],'w');
fprintf(fid,'%g %g\n', nz);
fclose(fid);
save(['z.dat'],'Z','-ascii','-append');

% running the executable and timing
tic
system(['./arma_model ' num2str(nstart-1) ' ' num2str(num_ts) ' ' num2str(num_forecast) ' ' num2str(nce) ' ' num2str(xv) ' z.dat ' num2str(adap)]);
toc
disp('Finished building model');

% reading results
load z.dat;
nz = size(z,2);
load forecasts.dat;
nf = size(forecasts,2);

% plotting the base coefficients 
colors = [102,194,165
252,141,98
141,160,203
231,138,195
166,216,84
255,217,47
229,196,148]/255;

lwidth = 4;
outwidth = 3;
fsize = 24;

h1011 = figure('Units','Pixels','Position',[0,0,800,600],'PaperPositionMode','auto','InvertHardCopy','off','Color','w');

hold on;
for i = 1:size(z,1)%num_nz_coeff_eff
    plot(z(i,:),'linewidth',outwidth,'color',colors(mod(i,7)+1,:));
    plot(nz-nf+[1:nf],forecasts(i,:),'o','MarkerSize',10,'MarkerFaceColor','w','MarkerEdgeColor',colors(mod(i,7)+1,:),'color',colors(mod(i,7)+1,:),'linewidth',2);
end

title ('True and forecast wavelet coefficients','FontName','Times','FontSize',fsize,'FontWeight','bold');
xlabel('Time','FontName','Times','FontSize',fsize,'FontWeight','bold');
ylabel('wavelet coefficients','FontName','Times','FontSize',fsize,'FontWeight','bold');
legend('Truth','forecast','Location','northeast');
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
print('-depsc2','-r300','-loose','wc.eps');

% loading the true solutions and the wavelet matrix
load(matfile,'u','V');
load V.mat

% reconstructing solutions and computing errors
for i=1:num_forecast
    f=u(:,num_ts+nstart-1+i); % subset of the true solution

    % 'es' is initial error due to lossy compression
    a=wc_timeseries(:,num_ts+nstart-1+i);
    es(i)=sqrt(sum((f-V*a).^2)/length(f))/(max(f)-min(f));
    
    % 'es_nce' is initial error due to lossy compression using nce
    % coefficients only
    [as,ii]=sort(abs(a),'descend');
    a(ii(nce+1:end))=0;
    es_nce(i)=sqrt(sum((f-V*a).^2)/length(f))/(max(f)-min(f));
    
    % 'est' is the compound error using nce coefficients and forecasting
    a=zeros(size(wc_timeseries,1),1);      % vector of forecast wavelet transform 
    for j=1:nce
        a(iw_cs(j))=forecasts(j,i);
    end
    est(i)=sqrt(sum((f-V*a).^2)/length(f))/(max(f)-min(f));
end

% plotting the original and compund errors
h1011 = figure('Units','Pixels','Position',[0,0,800,600],'PaperPositionMode','auto','InvertHardCopy','off','Color','w');
hold on;
plot(1:num_forecast,es,'linewidth',outwidth,'color',colors(mod(1,7)+1,:))
plot(1:num_forecast,es_nce,'linewidth',outwidth,'color',colors(mod(2,7)+1,:))
plot(1:num_forecast,est,'linewidth',outwidth,'color',colors(mod(3,7)+1,:))
title('Forecast error metrics after inverse transformation','FontName','Times','FontSize',fsize,'FontWeight','bold');
xlabel('Forecast time','FontName','Times','FontSize',fsize,'FontWeight','bold');
ylabel('Normalized L_2 error in the data field','FontName','Times','FontSize',fsize,'FontWeight','bold');
ll = legend('Minimum error due to lossy compression',['Error due to lossy compression using ' num2str(nce) ' coefficients'],['Increased error due to AR forecasting using ' num2str(nce) ' coefficients']);
ll.Location='northwest';
ll.FontSize = fsize-4;
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
print('-depsc2','-r300','-loose','errors.eps');


% reading mesh
load mesh_toy.mat
x=p(1,:);
y=p(2,:);

% plotting reconstructed results
% i is the index of the time step to be forecast 1 <= i <= num_forecast 
i=10;
f=u(:,num_ts+nstart-1+i);
% 'fs' is the full most accurate reconstruction
a=wc_timeseries(:,num_ts+nstart-1+i);
fs=V*a;
[as,ii]=sort(abs(a),'descend');
% 'fs_nce' is the reconstruction using nce coefficients
a(ii(nce+1:end))=0;
fs_nce=V*a;
% 'fst' is the reconstruction after forecasting using nce coefficients
a=zeros(size(wc_timeseries,1),1);      % vector of forecast wavelet transform 
for j=1:nce
    a(iw_cs(j))=forecasts(j,i);
end
fst=V*a;

fsize = 16;

plotr=[min(f) max(f)];
h1011 = figure('Units','Pixels','Position',[0,0,800,600],'PaperPositionMode','auto','InvertHardCopy','off','Color','w');
subplot(2,2,1)
trisurf(t(1:3,:)',x,y,f);view(2);shading interp
title('Original','FontName','Times','FontSize',fsize,'FontWeight','bold');
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
caxis(plotr);
subplot(2,2,2)
trisurf(t(1:3,:)',x,y,fs);view(2);shading interp
title(['Reconstructed after lossy compression' char(10) 'using all wavelet coefficients'],'FontName','Times','FontSize',fsize,'FontWeight','bold');
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
caxis(plotr);
subplot(2,2,3)
trisurf(t(1:3,:)',x,y,fs_nce);view(2);shading interp
title(['Reconstructed after lossy compression' char(10) 'using 60 wavelet coefficients'],'FontName','Times','FontSize',fsize,'FontWeight','bold');
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
caxis(plotr);
subplot(2,2,4)
trisurf(t(1:3,:)',x,y,fst);view(2);shading interp
caxis(plotr);
title(['Reconstructed after lossy compression' char(10) 'using 60 (AR) forecasted wavelet coefficients'],'FontName','Times','FontSize',fsize,'FontWeight','bold');
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
print('-depsc2','-r300','-loose','field.eps');
