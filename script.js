document.addEventListener('DOMContentLoaded', () => {
    const locationSelect = document.getElementById('location');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');

    const amenityColumns = [
        'Resale', 'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
        'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom',
        'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup',
        'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital',
        'WashingMachine', 'Gasconnection', 'AC', 'Wifi', "Children'splayarea",
        'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
        'DiningTable', 'Sofa', 'Wardrobe', 'Stadium'
    ];

    // Function to populate locations dropdown
    async function populateLocations() {
        try {
            // Fetch from your Flask endpoint '/get_locations'
            const response = await fetch('http://127.0.0.1:5000/get_locations');
            const data = await response.json(); // Flask returns { "locations": [...] }

            if (data.error) {
                console.error("Error fetching locations:", data.error);
                alert('Error loading locations: ' + data.error);
                return;
            }

            const locations = data.locations; // Access the 'locations' array from the response

            // Add a default Select a location option
            const defaultOption = document.createElement('option');
            defaultOption.value = "";
            defaultOption.textContent = "Select a location";
            defaultOption.disabled = true;
            defaultOption.selected = true;
            locationSelect.appendChild(defaultOption);

            locations.forEach(location => {
                const option = document.createElement('option');
                option.value = location;
                option.textContent = location;
                locationSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error fetching locations:', error);
            alert('Could not load locations. Please ensure the backend server is running.');
        }
    }

    // Call function to populate locations on page load
    populateLocations();

    predictButton.addEventListener('click', async () => {
        const area = parseFloat(document.getElementById('area').value);
        const bedrooms = parseInt(document.getElementById('bedrooms').value);
        const location = locationSelect.value;

        // Basic validation
        if (isNaN(area) || area <= 0) {
            alert('Please enter a valid Area (positive number).');
            return;
        }
        if (isNaN(bedrooms) || bedrooms <= 0) {
            alert('Please enter a valid number of Bedrooms (positive integer).');
            return;
        }
        if (!location) {
            alert('Please select a Location.');
            return;
        }

        const data = {
            "Area": area,
            "No. of Bedrooms": bedrooms,
            "Location": location
        };

        amenityColumns.forEach(col => {
            const checkbox = document.getElementById(col);
            data[col] = checkbox && checkbox.checked ? 1 : 0;
        });

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong on the server.');
            }

            const result = await response.json();
            predictionResult.textContent = `Predicted Price: ₹${result.predicted_price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
            predictionResult.style.display = 'block'; // Show the result
            predictionResult.style.backgroundColor = '#e9f7ef'; // Success background
            predictionResult.style.color = '#28a745'; // Success text color
            predictionResult.style.borderColor = '#c3e6cb'; // Success border
        } catch (error) {
            console.error('Error:', error);
            predictionResult.textContent = `Error: ${error.message}`;
            predictionResult.style.backgroundColor = '#f8d7da'; // Error background
            predictionResult.style.color = '#721c24'; // Error text color
            predictionResult.style.borderColor = '#f5c6cb'; // Error border
            predictionResult.style.display = 'block';
        }
    });
});