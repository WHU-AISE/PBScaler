package travel2.entity;

import lombok.Data;
import org.springframework.data.annotation.Id;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TripResponse {
    @Valid
    @Id
    private TripId tripId;

    @Valid
    @NotNull
    private String trainTypeId;

    @Valid
    @NotNull
    private String startingStation;

    @Valid
    @NotNull
    private String terminalStation;

    @Valid
    @NotNull
    private Date startingTime;

    @Valid
    @NotNull
    private Date endTime;

    /**
     * the number of economy seats
     */
    @Valid
    @NotNull
    private int economyClass;

    /**
     * the number of confort seats
     */
    @Valid
    @NotNull
    private int confortClass;

    @Valid
    @NotNull
    private String priceForEconomyClass;

    @Valid
    @NotNull
    private String priceForConfortClass;

    public TripResponse(){
        //Default Constructor
        this.trainTypeId = "";
        this.startingStation = "";
        this.terminalStation = "";
        this.startingTime = new Date();
        this.endTime = new Date();
        this.economyClass = 0;
        this.confortClass = 0;
        this.priceForEconomyClass = "";
        this.priceForConfortClass = "";
    }

}
