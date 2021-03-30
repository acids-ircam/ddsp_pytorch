% @file
% @ingroup 	jamoma2
% 
% @brief 		Generates the Expected Target Output for Interpolators using Octave
%
% @author		Nathan Wolek
% @copyright	Copyright (c) 2005-2015 The Jamoma Group, http://jamoma.org.
% @license		This project is released under the terms of the MIT License.

clear

% starting values
x0 = -1.0;
x1 =  2.0;
x2 =  1.0;
x3 =  4.0;
x4 =  3.0;
x = [x0,x1,x2,x3,x4];

output_linear = double (1 : 64);
output_hermite = double (1 : 64);
output_hermitegen = double (1 : 64);
output_spline = double (1 : 64);
output_splinegen = double (1 : 64);
output_cubicgen = double (1 : 64);
output_cosinegen = double (1 : 64);
output_allpassgen = double (1 : 64);
lastout_allpass = 0.0;

% the following function is adapted from gen~.interpolation example from Max 7.1
function retval = interp_hermitegen(v,delta)
	retval = 0.0;
	bias = 0.5;
	tension = 0.5;
	delta_int = fix(delta);
	a = delta - delta_int;
	w = v(delta_int-1);
	x = v(delta_int);
	y = v(delta_int+1);
	z = v(delta_int+2);
	aa = a*a;
	aaa = a*aa;
	bp = 1+bias;
	bm = 1-bias;
	mt = (1-tension)*0.5;
	m0  = ((x-w)*bp + (y-x)*bm) * mt;
	m1  = ((y-x)*bp + (z-y)*bm) * mt;
	a0 =  2*aaa - 3*aa + 1;
   	a1 =    aaa - 2*aa + a;
   	a2 =    aaa -   aa;
   	a3 = -2*aaa + 3*aa;
   	retval = a0*x + a1*m0 + a2*m1 + a3*y;
endfunction

% the following function is adapted from gen~.interpolation example from Max 7.1
function retval = interp_splinegen(v,delta)
	retval = 0.0;
	delta_int = fix(delta);
	a = delta - delta_int;
	w = v(delta_int-1);
	x = v(delta_int);
	y = v(delta_int+1);
	z = v(delta_int+2);
	a2 = a*a;
	f0 = -0.5*w + 1.5*x - 1.5*y + 0.5*z;
	f1 = w - 2.5*x + 2*y - 0.5*z;
	f2 = -0.5*w + 0.5*y;
	retval = f0*a*a2 + f1*a2 + f2*a + x;
endfunction

% the following function is adapted from gen~.interpolation example from Max 7.1
function retval = interp_cubicgen(v,delta)
	retval = 0.0;
	delta_int = fix(delta);
	a = delta - delta_int;
	w = v(delta_int-1);
	x = v(delta_int);
	y = v(delta_int+1);
	z = v(delta_int+2);
	a2 = a*a;
	f0 = z - y - w + x;
	f1 = w - x - f0;
	f2 = y - w;
	retval = f0*a*a2 + f1*a2 + f2*a + x;
endfunction

% the following function is adapted from gen~.interpolation example from Max 7.1
function retval = interp_cosinegen(v,delta)
	retval = 0.0;
	delta_int = fix(delta);
	a = delta - delta_int;
	x = v(delta_int);
	y = v(delta_int+1);
	a2 = 0.5*(1-cos(a*pi));
	retval = x + a2*(y-x);
endfunction

% reference: https://ccrma.stanford.edu/~jos/pasp/First_Order_Allpass_Interpolation.html
function retval = interp_allpassgen(v,delta,history)
	retval = 0.0;
	delta_int = fix(delta);
	a = delta - delta_int;
	% the following if statement corrects for a difference between deltas of 0.0 and 1.0 in this algorithm
	if (a == 0.0)
		delta_int = delta_int - 1;
		a = 1.0;
	endif
	x1 = v(delta_int);
	x2 = v(delta_int+1);
	out = x1 + a*(x2-history);
	retval = out;
endfunction

for i = 1:64
	current_delta = 2.0 + i / 64;
	output_linear(i) = interp1(x,current_delta);
	output_hermite(i) = interp1(x,current_delta,"pchip");
	output_hermitegen(i) = interp_hermitegen(x,current_delta);
	output_spline(i) = interp1(x,current_delta,"spline");
	output_splinegen(i) = interp_splinegen(x,current_delta);
	output_cubicgen(i) = interp_cubicgen(x,current_delta);
	output_cosinegen(i) = interp_cosinegen(x,current_delta);
	output_allpassgen(i) = interp_allpassgen(x,current_delta,lastout_allpass);
	lastout_allpass = output_allpassgen(i);
endfor

save expectedOutput.mat output_linear
save -append expectedOutput.mat output_hermite
save -append expectedOutput.mat output_hermitegen
save -append expectedOutput.mat output_spline
save -append expectedOutput.mat output_splinegen
save -append expectedOutput.mat output_cubicgen
save -append expectedOutput.mat output_cosinegen
save -append expectedOutput.mat output_allpassgen