echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 53780 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 25760) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 30616) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 28500) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 26924) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 29476) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 20340)
del "C:\Users\Research\Desktop\RA\Code\CoreTempAI\cases\general\cleanup-fluent-BME-BANER-01-29476.bat"
