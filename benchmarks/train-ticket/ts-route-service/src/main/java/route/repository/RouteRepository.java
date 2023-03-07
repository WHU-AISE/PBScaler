package route.repository;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import route.entity.Route;
import java.util.ArrayList;

/**
 * @author fdse
 */
@Repository
public interface RouteRepository extends MongoRepository<Route, String> {

    /**
     * find route by id
     *
     * @param id id
     * @return Route
     */
    @Query("{ 'id': ?0 }")
    Route findById(String id);

    /**
     * find all routes
     *
     * @return ArrayList<Route>
     */
    @Override
    ArrayList<Route> findAll();

    /**
     * remove route via id
     *
     * @param id id
     */
    void removeRouteById(String id);

    /**
     * return route with id from StartStationId to TerminalStationId
     *
     * @param startingId  Start Station Id
     * @param terminalId  Terminal Station Id
     * @return ArrayList<Route>
     */
    @Query("{ 'startStationId': ?0 , 'terminalStationId': ?1 }")
    ArrayList<Route> findByStartStationIdAndTerminalStationId(String startingId, String terminalId);

}
