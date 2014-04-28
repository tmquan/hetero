function u = focuss_20d(R, f, params)
	%%  R       sampling R
	%%  f       undersampled kspace
	%%  Mu      Mu
	%%  Lambda  Lambda
	%%  gamma   Gamma
	%%  nInner  number of inner loops
	%%  nOuter  number of outer loops

    if(~exist('params'))
        % Bregman Parameters
        params.Mu      	= 0.1;
        params.Lambda	= 0.01; 
        params.Gamma   	= 0.5;
        params.Ep 		= 0.7;
        params.nOuter  	= 4;
        params.nInner  	= 8;
        params.nLoops  	= 5;
    end
    
    Mu      = params.Mu;
    Lambda  = params.Lambda;
    Gamma   = params.Gamma;
    Ep      = params.Ep;
    nOuter  = params.nOuter;
    nInner  = params.nInner;
    nLoops  = params.nLoops;
    
    factor = 0.5;
    %%
    f0 = f;
	%% Get size of the kspace  in dimx dimy and temporals 
	dim = size(f);

	u0 = abs(ifft2(f0));
    %% Construct the w matrix
    w = abs(u0).^factor;
    w = w./max(w(:));

    %% Iteratively reconstruct the images
    tStart = tic;
    for iOuter = 1:nOuter
        p = zeros(size(w));
        x = zeros(dim);
        for iInner = 1:nInner
            fprintf('Outer loop %dth, Inner loop %dth\n', iOuter, iInner);
            
            x = fft(x,[],3);    % Make the signal sparser along temporal dimension
            x = fft2(x);        % Precompute before conjugate g
            r = f0 - R.*x;
            r = r.*R;

            g = ifft2(r);
            g = ifft(g,[],3);
            g = -w.*g;     
            %% Conjugate Gradient
            if(iInner>1)
                beta = g(:)'*g(:)/(prev_g(:)'*prev_g(:));
                d = -g+beta*prev_d;
            else
                d = -g;
            end
            prev_d  = d;
            prev_g  = g;
            %%

            z = w.*d;
            z = fft(z,[],3);
            z = fft2(z);
            z = R.*z;
            alpha = (r(:)'*z(:)-Lambda*(d(:)'*p(:))) ...
                     /(z(:)'*z(:)+Lambda*d(:)'*d(:));
            p = p+alpha*d;
            x = w.*p;
            %%
            u = x;
            u = ifft(u,[],3);
            imagesc(abs(u)); colormap gray; axis square off; drawnow;
        end
        w = abs(x).^factor;    
        w = w/max(w(:));      
    end
    u = ifft(x,[],3);
	tEnd = toc(tStart);
	fprintf('%d minutes and %f seconds\n',floor(tEnd/60),rem(tEnd,60));
return;


% !!! Central differences
function d = Dx(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,2:cols-1,:) = (u(:,3:cols,:)-u(:,1:cols-2,:))/2.0;
	d(:,1,:) = u(:,2,:)-u(:,1,:); % No wrap-around
	d(:,cols,:) = u(:,cols,:)-u(:,cols-1,:); % No wrap-around
return

% !!!
function d = Dxt(u)
	d = -Dx(u);
return

function d = Dy(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(2:rows-1,:,:) = (u(3:rows,:,:)-u(1:rows-2,:,:))/2.0;
	d(1,:,:) = u(2,:,:)-u(1,:,:);
	d(rows,:,:) = u(rows,:,:)-u(rows-1,:,:);
return

function d = Dyt(u)
	d = -Dy(u);
return


function d = Dz(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,:,2:tems-1) = (u(:,:,3:tems)-u(:,:,1:tems-2))/2.0;
	d(:,:,1) = u(:,:,2)-u(:,:,1); % No wrap-around
	d(:,:,tems) = u(:,:,tems)-u(:,:,tems-1); % No wrap-around
return

% !!!
function d = Dzt(u)
d = -Dz(u);
return

% !!! Laplacian along x d
function d = Lx(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,2:cols-1,:) = u(:,1:cols-2,:)-2*u(:,2:cols-1,:)+u(:,3:cols,:);
	d(:,1,:) = u(:,2,:)-u(:,1,:);
	d(:,cols,:) = u(:,cols-1,:)-u(:,cols,:);
	d = -1*d;
return

% !!! Laplacian along y d
function d = Ly(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(2:rows-1,:,:) = u(1:rows-2,:,:)-2*u(2:rows-1,:,:)+u(3:rows,:,:);
	d(1,:,:) = u(2,:,:)-u(1,:,:);
	d(rows,:,:) = u(rows-1,:,:)-u(rows,:,:);
	d = -1*d;
return

% !!! Laplacian along z d
function d = Lz(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,:,2:tems-1) = u(:,:,1:tems-2)-2*u(:,:,2:tems-1)+u(:,:,3:tems);
	d(:,:,1) = u(:,:,2)-u(:,:,1);
	d(:,:,tems) = u(:,:,tems-1)-u(:,:,tems);
	d = -1*d;
return



function [xs,ys,zs] = shrink3(x,y,z,Lambda)
	s = sqrt(x.*conj(x)+y.*conj(y)+z.*conj(z));
	ss = s-Lambda;
	ss = ss.*(ss>0);

	% s = s+(s<Lambda);
	ss = ss./s;

	xs = ss.*x;
	ys = ss.*y;
	zs = ss.*z;
return;
%%
function [xs,ys] = shrink2(x,y,Lambda)
	s = sqrt(x.*conj(x)+y.*conj(y));
	ss = s-Lambda;
	ss = ss.*(ss>0);

	% s = s+(s<Lambda);
	ss = ss./s;

	xs = ss.*x;
	ys = ss.*y;
return;

function [xs] = shrink1(x,Lambda)
	s = sqrt(x.*conj(x));
	ss = s-Lambda;
	ss = ss.*(ss>0);

	% s = s+(s<Lambda);
	ss = ss./s;

	xs = ss.*x;
return;

function y =  print(x);
	[rows,cols,tems] = size(x);
	fprintf('===========================================================');
	x(1,1,1)
	x(2,1,1)
	x(3,1,1)
	x(4,1,1)
	% x(rows,cols,tems)
	% fprintf('%f \n', x(1,1,1));
	% fprintf('%f \n', x(2,1,1));
	% fprintf('%f \n', x(3,1,1));
	% fprintf('%f \n', x(rows, cols, tems));
	% fprintf('%e \t %e\n', real(x(rows, cols, tems)), imag(x(rows, cols, tems)));
	% fprintf('%e \t %e\n', real(x(1, 1, 1)), imag(x(1, 1, 1)));
	% fprintf('%e \t %e\n', real(x(2, 1, 1)), imag(x(2, 1, 1)));
	% fprintf('%e \t %e\n', real(x(3, 1, 1)), imag(x(3, 1, 1)));
	% fprintf('%e \t %e\n', real(x(4, 1, 1)), imag(x(4, 1, 1)));
	% fprintf('%e \t %e\n', real(x(rows, cols, tems)), imag(x(rows, cols, tems)));
return;
