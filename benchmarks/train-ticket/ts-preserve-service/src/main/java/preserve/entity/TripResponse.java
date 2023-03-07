package preserve.entity;

import lombok.Data;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TripResponse {
    @Valid
    private TripId tripId;

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
     * the number of economy seat
     */
    @Valid
    @NotNull
    private int economyClass;

    /**
     * the number of confort seat
     */
    @Valid
    @NotNull
    private int confortClass;

}
