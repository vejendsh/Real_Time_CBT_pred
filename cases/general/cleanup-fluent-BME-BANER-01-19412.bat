echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 53201 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 10856) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 1288) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 17704) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 28608) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 19412) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 28580)
del "C:\Users\Research\Desktop\RA\Code\CoreTempAI\cases\general\cleanup-fluent-BME-BANER-01-19412.bat"
