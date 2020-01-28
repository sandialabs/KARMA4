% run_test - MATLAB script to generate time-series data and subsequently
% utilize karma4 library for autoregressive model parameter calibration and
% order selection
%
% Authors: Moe Khalil, Jina Lee, Maher Salloum
% Sandia National Labopratories
% email: mkhalil@sandia.gov
% Website: http://www.sandia.gov/~mnsallo/software.html
% August 2017; Last revision: 15-August-2017


% Cleaning environment
clear all; close all;

% Resetting state of random number generator: needed for reproducibility
randn('state',1);

% Choice of generating model
% gen_model = 0 for AR model
% gen_model = 1 for exponentially growing sinusoidal data
gen_model = 1;

nz = 100; % number of data points
gamma = 0.01; %measurement noise intensity (standard deviation)

% generate clean time-series data
switch gen_model
    case 0
        
        init_z = [0.889178995665789   1.0]';

        phi = [0.25 0.72];
        nphi = length(phi);

        n = nz-length(init_z);
        sigma = 0.01; %model noise intensity (standard deviation)

        x = init_z;
        for i = 1:n
            x = [phi*x;x(1:(end-1))] + sigma*randn;
            y(i) = x(1);
        end
        y = [flipud(init_z);y(:)];
        
    case 1
        
        y = exp(1.0e-2*(1:nz)).*sin(0.2*(1:nz));
end

% generate noisy (Gaussian) observations of time-series
y = y + gamma*randn(size(y));

% save relevant data to file for loading by arma_model app
fid=fopen('z.dat','w');
fprintf(fid, '%e\n', y);
fclose(fid);
fileID = fopen('nz.dat','w');
fprintf(fileID,'%d\n',nz);
fclose(fileID);

% Building the model using arma_model driving application and karma4
% library
system('./arma_model');

% load and plot noisy observations and arma model forecasts
load z.dat;
nz = length(z);


fsize = 30;
lwidth = 4;
outwidth = 2;

figure('Units','Pixels','Position',[0,0,1100,700],'PaperPositionMode','auto','InvertHardCopy','off','Color','w');
hold on;
plot(z,'b','linewidth',3);

load forecasts.dat
plot((length(z)+1:1:length(z)+length(forecasts)),forecasts,'o-','MarkerSize',6,'MarkerFaceColor',[0 0.6 0],'MarkerEdgeColor',[0 0.6 0],'color',[0 0.6 0],'linewidth',2)

load forecast_err.dat
plot((length(z)+1:1:length(z)+length(forecasts)),forecasts-3*sqrt(forecast_err),'-','color',[0 0.6 0],'linewidth',2)
plot((length(z)+1:1:length(z)+length(forecasts)),forecasts+3*sqrt(forecast_err),'-','color',[0 0.6 0],'linewidth',2)


legendsi{1} = 'Noisy data';
legendsi{2} = 'Forecast (mean)';
legendsi{3} = 'Forecast confidence interval';

axis tight;
temp1 = axis;
axis(temp1);
xlabel('time','FontName','Times','FontSize',fsize,'FontWeight','bold');
ylabel('z_t','FontName','Times','FontSize',fsize,'FontWeight','bold','Rotation',0);
legend(legendsi,'Location','southwest');
set(gca,'FontName','Times','FontSize',fsize,'linewidth',outwidth,'FontWeight','bold')
print('-depsc2','-r300','-loose','ar_valid.eps');
