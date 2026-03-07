echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 55027 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 28368) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 29704) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 10124) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 7300) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 26916) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 30380)
del "C:\Users\Research\Desktop\RA\Code\CoreTempAI\cases\general_st_indep\cleanup-fluent-BME-BANER-01-26916.bat"
