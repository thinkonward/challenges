# Set your AWS S3 details
$sourceFolder = "oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
# Define the different release versions you want to iterate over
$releases = @("2024/comstock_amy2018_release_1", "2022/resstock_amy2018_release_1.1")
$folderToData = "/timeseries_individual_buildings/by_state/"
$destinationFolder = "data_dump/"
$stateList = @("AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY")

& ".venv\Scripts\activate"

# Set the number of files you want to copy
$N = [int](Read-Host "Enter Number of Home to scrape (-1 if no random extraction)")

if ($N -gt 0) {
	$EstimatedNBuilding = $N*2*51
	Write-Host "Estimated number of new buildings: $EstimatedNBuilding"	
}
else {
	Write-Host "No sampling, taking every data"
}
#reset log
"" | Out-File -FilePath "log/dumping.txt"


foreach ($release in $releases){
	
	Write-Host "Starting $release"
	"Starting $release" | Out-File -FilePath "log/dumping.txt" -Append

	#define utils byt type
	if ($release -eq "2024/comstock_amy2018_release_1"){
		$upgradeId = "upgrade=32/"
		$typeBuildFolderName = "commercial"
	}
	else {
		$upgradeId = "upgrade=10/"
		$typeBuildFolderName = "residential"
	}
	
	# List and sort the objects in the source folder
	$currentBucket = "s3://$sourceFolder$release$folderToData$upgradeId"
	$Folders = aws s3 ls $currentBucket --no-sign-request | Sort-Object

	foreach ($folder in $Folders) {
		#find state name
		$folderName = ($folder -split '\s+')[-1]
		$stateName = ($folderName -split '=')[-1] -replace '/', ''

		$pythonCommand = "script/clean_data.py --state=$stateName --type_building=$typeBuildFolderName"

		#they put some wrong state as NA
		if ($stateName -notin $stateList){
			Write-Host "Skipping $stateName as not a US State"
			"Skipping $stateName as not a US State" | Out-File -FilePath "log/dumping.txt" -Append
			continue
		}

		$localFolderPath = "$destinationFolder$typeBuildFolderName/$folderName"
		
		Write-Host "Starting $folderName"
		"Starting $folderName" | Out-File -FilePath "log/dumping.txt" -Append

		If (!(test-path $localFolderPath)){
			mkdir $localFolderPath
		}
		$fileList = aws s3 ls "$currentBucket$folderName" --no-sign-request
		
		if ($N -gt 0) {
			#list all and get N random
			$fileList = $fileList | Sort-Object { Get-Random }
					
			# Select the first N objects
			$fileList = $fileList | Select-Object -First $N
		
			#define starting command
			$awsBaseCommand = "aws s3 cp `"$currentBucket$folderName`" $localFolderPath --recursive --no-sign-request --exclude `"*`""
			
			$numFile = 0
			$awsCommand = $awsBaseCommand
			$missingNumFile = $fileList.Count

			Write-Host "$missingNumFile total number of file to download"
			"$missingNumFile total number of file to download" | Out-File -FilePath "log/dumping.txt" -Append

			#add every include file in bacth of 1000 file each
			foreach ($file in $fileList) {
				$fileName = ($file -split '\s+')[-1]
				$awsCommand += " --include `"$fileName`""
				$numFile = $numFile + 1

				if ($numFile -eq 1000){
					#download every selected file
					Try {
						Invoke-Expression $awsCommand
						Start-Sleep 1	
					} Catch {
						Write-Host "Caught an error: $_"
					}
					
					$numFile = 0
					$missingNumFile -= 1000
					$awsCommand = $awsBaseCommand

					Write-Host "Remaining $missingNumFile file"
					"Remaining $missingNumFile file" | Out-File -FilePath "log/dumping.txt" -Append
				}
			}
			if($numFile -gt 0){
				Invoke-Expression $awsCommand
			}
		}
		else{
			$missingNumFile = $fileList.Count

			Write-Host "$missingNumFile total number of file to download"
			"$missingNumFile total number of file to download" | Out-File -FilePath "log/dumping.txt" -Append

			$awsCommand = "aws s3 cp `"$currentBucket$folderName`" $localFolderPath --recursive --no-sign-request --cli-read-timeout 0 --cli-connect-timeout 0"

			Try {
				Invoke-Expression $awsCommand
				Start-Sleep 1	
			} Catch {
				Write-Host "Caught an error: $_"
			}
		}

		Write-Host "done"
		"done" | Out-File -FilePath "log/dumping.txt" -Append

		#read and concat every file, after that delete every chunk file
		Write-Host "reducing file size and deleting multiple dataset"
		"reducing file size and deleting multiple dataset" | Out-File -FilePath "log/dumping.txt" -Append
		Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $pythonCommand -NoNewWindow -Wait

		Write-Host "done"
		"done" | Out-File -FilePath "log/dumping.txt" -Append
		Start-Sleep -Seconds 60
	}
}
