% @file
% @ingroup 	minlib
% 
% @brief 		Generates the Expected Target Output for Easing using Octave
%
% @author		Nathan Wolek
% @copyright	Copyright (c) 2017 Nathan Wolek
% @license		This project is released under the terms of the MIT License.

clear

% NOTE: in developing the code for this file, I consulted the following:
% https://github.com/warrenm/AHEasing/blob/master/AHEasing/easing.c
% https://github.com/Cycling74/min-lib/blob/master/include/c74_lib_easing.h

% Variables at the top are used to vary the target outputs.
% - samples_to_output controls the number of samples in each target output matrix.
samples_to_output = 64;

% 1 - initialize all matrices to hold values generated
input_ramp = double (1 : samples_to_output);
output_linear = double (1 : samples_to_output);
output_in_back = double (1 : samples_to_output);
output_in_out_back = double (1 : samples_to_output);
output_out_back = double (1 : samples_to_output);
output_out_bounce = double (1 : samples_to_output);
output_in_bounce = double (1 : samples_to_output);
output_in_out_bounce = double (1 : samples_to_output);
output_in_circular = double (1 : samples_to_output);
output_out_circular = double (1 : samples_to_output);
output_in_out_circular = double (1 : samples_to_output);
output_in_cubic = double (1 : samples_to_output);
output_in_out_cubic = double (1 : samples_to_output);
output_out_cubic = double (1 : samples_to_output);
output_in_elastic = double (1 : samples_to_output);
output_out_elastic = double (1 : samples_to_output);
output_in_out_elastic = double (1 : samples_to_output);
output_in_exponential = double (1 : samples_to_output);
output_out_exponential = double (1 : samples_to_output);
output_in_out_exponential = double (1 : samples_to_output);
output_in_quadratic = double (1 : samples_to_output);
output_out_quadratic = double (1 : samples_to_output);
output_in_out_quadratic = double (1 : samples_to_output);
output_in_quartic = double (1 : samples_to_output);
output_out_quartic = double (1 : samples_to_output);
output_in_out_quartic = double (1 : samples_to_output);
output_in_quintic = double (1 : samples_to_output);
output_out_quintic = double (1 : samples_to_output);
output_in_out_quintic = double (1 : samples_to_output);
output_in_sine = double (1 : samples_to_output);
output_out_sine = double (1 : samples_to_output);
output_in_out_sine = double (1 : samples_to_output);

% 2 - define any functions used to generate values
function retval = in_out_back(inval)
	retval = 0.0; 
	if (inval < 0.5)
		f = 2 * inval;
		retval = 0.5 * (f * f * f - f * sin(f * pi));
	else
		f = (1 - (2*inval - 1));
		retval = 0.5 * (1 - (f * f * f - f * sin(f * pi))) + 0.5;
	endif
endfunction

function retval = out_back(inval)
	retval = 0.0;
	f = (1 - inval);
	retval = 1 - (f * f * f - f * sin(f * pi));
endfunction

function retval = out_bounce(inval)
	retval = 0.0;
	if (inval < 4/11.0)
		retval = (121 * inval * inval)/16.0;
	elseif (inval < 8/11.0)
		retval = (363/40.0 * inval * inval) - (99/10.0 * inval) + 17/5.0;
	elseif(inval < 9/10.0)
		retval = (4356/361.0 * inval * inval) - (35442/1805.0 * inval) + 16061/1805.0;
	else
		retval = (54/5.0 * inval * inval) - (513/25.0 * inval) + 268/25.0;
	endif
endfunction

function retval = in_bounce(inval)
	retval = 1 - out_bounce(1 - inval);;
endfunction

function retval = in_out_bounce(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 0.5 * in_bounce(inval*2);
	else
		retval = 0.5 * out_bounce(inval * 2 - 1) + 0.5;
	endif
endfunction

function retval = in_out_circular(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 0.5 * (1 - sqrt(1 - 4 * (inval * inval)));
	else
		retval = 0.5 * (sqrt(-((2 * inval) - 3) * ((2 * inval) - 1)) + 1);
	endif
endfunction

function retval = in_out_cubic(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 4.0 * inval * inval * inval;
	else
		f = ((2 * inval) - 2);
		retval = 0.5 * f * f * f + 1;
	endif
endfunction

function retval = out_cubic(inval)
	retval = 0.0;
	f = inval - 1.0;
	retval = f * f * f + 1.0;
endfunction

function retval = in_out_elastic(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 0.5 * sin(6.5 * pi * (2 * inval)) * 2^(10 * ((2 * inval) - 1));
	else
		retval = 0.5 * (sin(-6.5 * pi * ((2 * inval - 1) + 1)) * 2^(-10 * (2 * inval - 1)) + 2);
	endif
endfunction

function retval = in_exponential(inval)
	retval = 0.0;
	if (inval == 0.0)
		retval = 0.0;
	else
		retval = 2^(10 * (inval - 1));
	endif
endfunction

function retval = out_exponential(inval)
	retval = 0.0;
	if (inval == 1.0)
		retval = 1.0;
	else
		retval = 1 - 2^(-10 * inval);
	endif
endfunction

function retval = in_out_exponential(inval)
	retval = 0.0;
	if (inval == 0.0)
		retval = 0.0;
	elseif (inval == 1.0)
		retval = 1.0;
	elseif (inval < 0.5)
		retval = 0.5 * 2^((20 * inval) - 10);
	else
		retval = -0.5 * 2^((-20 * inval) + 10) + 1;
	endif
endfunction

function retval = in_out_quadratic(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 2 * inval * inval;
	else
		retval = (-2 * inval * inval) + (4 * inval) - 1;
	endif
endfunction

function retval = out_quartic(inval)
	retval = 0.0;
	f = (inval - 1);
	retval = f * f * f * (1 - inval) + 1;
endfunction

function retval = in_out_quartic(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 8 * inval * inval * inval * inval;
	else
		f = (inval - 1);
		retval = -8 * f * f * f * f + 1;
	endif
endfunction

function retval = out_quintic(inval)
	retval = 0.0;
	f = (inval - 1);
	retval = f * f * f * f * f + 1;
endfunction

function retval = in_out_quintic(inval)
	retval = 0.0;
	if (inval < 0.5)
		retval = 16 * inval * inval * inval * inval * inval;
	else
		f = ((2 * inval) - 2);
		retval = 0.5 * f * f * f * f * f + 1;
	endif
endfunction

% 3 - iterate through loop to fill matrices
for i = 1:samples_to_output
	% NW: our formula for input_ramp is constructed so that 0 and 1 will be included
	x = (i-1) / (samples_to_output-1);
	input_ramp(i) = x;
	output_linear(i) = x;
	output_in_back(i) = x * x * x - x * sin(x * pi);
	output_in_out_back(i) = in_out_back(x);
	output_out_back(i) = out_back(x);
	output_out_bounce(i) = out_bounce(x);
	output_in_bounce(i) = in_bounce(x);
	output_in_out_bounce(i) = in_out_bounce(x);
	output_in_circular(i) = 1 - sqrt(1 - (x * x));
	output_out_circular(i) = sqrt((2 - x) * x);
	output_in_out_circular(i) = in_out_circular(x);
	output_in_cubic(i) = x * x * x;
	output_in_out_cubic(i) = in_out_cubic(x);
	output_out_cubic(i) = out_cubic(x);
	output_in_elastic(i) = sin(6.5 * pi * x) * 2^(10 * (x - 1));
	output_out_elastic(i) = sin(-6.5 * pi * (x + 1)) * 2^(-10 * x) + 1;
	output_in_out_elastic(i) = in_out_elastic(x);
	output_in_exponential(i) = in_exponential(x);
	output_out_exponential(i) = out_exponential(x);
	output_in_out_exponential(i) = in_out_exponential(x);
	output_in_quadratic(i) = x * x;
	output_out_quadratic(i) = -(x * (x - 2));
	output_in_out_quadratic(i) = in_out_quadratic(x);
	output_in_quartic(i) = x * x * x * x;
	output_out_quartic(i) = out_quartic(x);
	output_in_out_quartic(i) = in_out_quartic(x);
	output_in_quintic(i) = x * x * x * x * x;
	output_out_quintic(i) = out_quintic(x);
	output_in_out_quintic(i) = in_out_quintic(x);
	output_in_sine(i) = sin((x - 1) * pi * 0.5) + 1;
	output_out_sine(i) = sin(x * pi * 0.5);
	output_in_out_sine(i) = 0.5 * (1 - cos(x * pi));
endfor

% 4 - write output values to disk
save expectedOutput.mat input_ramp
save -append expectedOutput.mat output_linear
save -append expectedOutput.mat output_in_back
save -append expectedOutput.mat output_in_out_back
save -append expectedOutput.mat output_out_back
save -append expectedOutput.mat output_out_bounce
save -append expectedOutput.mat output_in_bounce
save -append expectedOutput.mat output_in_out_bounce
save -append expectedOutput.mat output_in_circular
save -append expectedOutput.mat output_out_circular
save -append expectedOutput.mat output_in_out_circular
save -append expectedOutput.mat output_in_cubic
save -append expectedOutput.mat output_in_out_cubic
save -append expectedOutput.mat output_out_cubic
save -append expectedOutput.mat output_in_elastic
save -append expectedOutput.mat output_out_elastic
save -append expectedOutput.mat output_in_out_elastic
save -append expectedOutput.mat output_in_exponential
save -append expectedOutput.mat output_out_exponential
save -append expectedOutput.mat output_in_out_exponential
save -append expectedOutput.mat output_in_quadratic
save -append expectedOutput.mat output_out_quadratic
save -append expectedOutput.mat output_in_out_quadratic
save -append expectedOutput.mat output_in_quartic
save -append expectedOutput.mat output_out_quartic
save -append expectedOutput.mat output_in_out_quartic
save -append expectedOutput.mat output_in_quintic
save -append expectedOutput.mat output_out_quintic
save -append expectedOutput.mat output_in_out_quintic
save -append expectedOutput.mat output_in_sine
save -append expectedOutput.mat output_out_sine
save -append expectedOutput.mat output_in_out_sine
