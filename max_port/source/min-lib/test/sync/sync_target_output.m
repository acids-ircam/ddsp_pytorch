% @file
% @ingroup 	jamoma2
% 
% @brief 		Generates the Expected Target Output for Sync using Octave
%
% @author		Nathan Wolek
% @copyright	Copyright (c) 2005-2015 The Jamoma Group, http://jamoma.org.
% @license		This project is released under the terms of the MIT License.

clear

% starting values
frequency = 3000.0;
initialPhase = 0.4;
sampleRate = 22050.0;

% algorithm variables
stepSize = frequency / sampleRate;
currentPhase = initialPhase - stepSize;

output = double (1 : 64);

for i = 1:64
	currentPhase = currentPhase + stepSize;
	if (currentPhase > 1.0)
		currentPhase = currentPhase - 1.0;
	elseif (currentPhase < 0.0)
		currentPhase = currentPhase + 1.0;
	endif
	output(i) = currentPhase;
endfor

save expectedOutput.mat output