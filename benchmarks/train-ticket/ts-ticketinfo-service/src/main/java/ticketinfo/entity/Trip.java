package ticketinfo.entity;

import lombok.Data;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import java.util.Date;

@Data
public class Trip {
    @Valid
    private TripId tripId;

    @Valid
    @NotNull
    private String trainTypeId;

    private String routeId;

    //-------------------------------The following variables are temporarily disabled-----------------------
    @Valid
    @NotNull
    private String startingStationId;

    //Intermediate stops. In the initial version, there was only one stop, that is, only one stop and one terminal. In the later version, the stops were expanded to several stops.
    @Valid
    private String stationsId;

    @Valid
    @NotNull
    private String terminalStationId;

    @Valid
    @NotNull
    private Date startingTime;

    @Valid
    @NotNull
    private Date endTime;

    public Trip(TripId tripId, String trainTypeId, String startingStationId, String stationsId, String terminalStationId, Date startingTime, Date endTime) {
        this.tripId = tripId;
        this.trainTypeId = trainTypeId;
        this.startingStationId = startingStationId;
        this.stationsId = stationsId;
        this.terminalStationId = terminalStationId;
        this.startingTime = startingTime;
        this.endTime = endTime;
    }

    public Trip(TripId tripId, String trainTypeId, String routeId) {
        this.tripId = tripId;
        this.trainTypeId = trainTypeId;
        this.routeId = routeId;
        this.startingStationId = "";
        this.terminalStationId = "";
        this.startingTime = new Date();
        this.endTime = new Date();
    }

    public Trip(){
        //Default Constructor
        this.trainTypeId = "";
        this.startingStationId = "";
        this.terminalStationId = "";
        this.startingTime = new Date();
        this.endTime = new Date();
    }

}
